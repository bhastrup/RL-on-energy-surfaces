
from datetime import datetime
import math
np.seterr(all='raise')
import random
from itertools import count
import logging
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax
import matplotlib
import matplotlib.pyplot as plt
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Import surf-rider functionality
from envs.ASE_rl_env import ASE_RL_Env
from models.random_agent import RandomAgent
from utils.memory import ReplayMemory
from utils.memory_mc import ReplayMemoryMonteCarlo
#from utils.slab_params import *
from utils.alloy import *
from utils.alloymap import AlloyGenerator
from utils.summary import PerformanceSummary
from utils.mirror import mirror
#from utils.neb import get_neb_energy

#from models.schnet_edge_model import SchnetModel
#from models.schnet_edge_model_action import SchnetModel
#from models.schnet_edge_model_action_no_int import SchnetModel
#from models.schnet_edge_model_action_no_cat import SchnetModel
#from models.schnet_edge_model_action_no_int_no_cat import SchnetModel
#import schnet_edge_model
from models import schnet_edge_model_action
#import schnet_edge_model_action_no_int
#import schnet_edge_model_action_no_cat
#import schnet_edge_model_action_no_int_no_cat
from models import data

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device==torch.device("cuda"):
    pin=True
else:
    pin=False

# Initialize reinforcement learning environment
env = ASE_RL_Env(
    initial_state=slab.copy(),
    goal_state=slab_b.copy(),
    agent_number=agent_atom,
    view=False,
    view_force=False
)

# Get number of actions from the environment class
n_actions = env.n_actions

# Random agent
k = 100 # softmax coefficient
sigma = 3 # exploration factor
agent = RandomAgent(action_space=env.action_space, k=k, sigma=sigma)


# Setup logging
now = datetime.now()
dt_string = now.strftime("%d_%m_%H_%M")
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, 'runs/defense/train_schnet_mirror' + dt_string + '/')
# output_dir = "runs/model_output"
# Setup logging
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(output_dir, "printlog.txt"), mode="w"
        ),
        logging.StreamHandler(),
    ],
)


def get_model(args, **kwargs):
    net = schnet_edge_model_action.SchnetModel(
        num_interactions=args.num_interactions,
        hidden_state_size=args.node_size,
        cutoff=args.cutoff,
        update_edges=args.update_edges,
        normalize_atomwise=args.atomwise_normalization,
        **kwargs
    )
    return net

# Hyper parameters
random_alloy = False
algorithm = "Monte-Carlo"
DOUBLE_Q = False
boltzmann = True
dublicate_mirror = True
mirror_action_space = np.array([0, 1, 3, 2, 4, 5])
T_BOLTZ = 0.25
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.1
EPS_END = 0.05
EPS_DECAY = 15000
TARGET_UPDATE = 10
Q_update = 3

num_episodes = 10000
num_episodes_train = 10
num_episodes_test  = 100

num_interactions = 3
node_size = 8
cutoff = 4.0 # assert that cutoff>np.cos(np.pi/6)*env.atom_object.get_cell().lengths()[0]
update_edges = True
atomwise_normalization = True
max_steps = 2000
learning_rate = 0.001

n_surf = - np.array([0.,0.,1.])

class args_wrapper():
    def __init__(self, num_interactions: int, node_size: int, cutoff: float, update_edges: bool, atomwise_normalization: bool, 
                 max_steps: int, device: torch.device, learning_rate: float, output_dir: str):
        self.num_interactions = num_interactions
        self.node_size = node_size
        self.cutoff = cutoff
        self.update_edges = update_edges
        self.atomwise_normalization = atomwise_normalization
        self.max_steps = max_steps
        self.device = device
        self.learning_rate = learning_rate
        self.output_dir = output_dir

args=args_wrapper(num_interactions, node_size, cutoff, update_edges, atomwise_normalization, max_steps, device, learning_rate, output_dir)
transformer = models.data.TransformAtomsObjectToGraph(cutoff=args.cutoff)

# Initialise model
if algorithm == "Q-learning":
    from utils.memory import ReplayMemory
    memory = ReplayMemory(10000)
    if DOUBLE_Q:
        netA = get_model(args).to(device)
        netA.eval()
        optimizerA = torch.optim.Adam(netA.parameters())
        netB = get_model(args).to(device)
        #netB.load_state_dict(netA.state_dict())
        netB.eval()
        optimizerB = torch.optim.Adam(netB.parameters())
    else:
        net = get_model(args).to(device)
        net.eval()
        target_net = get_model(args).to(device)
        target_net.load_state_dict(net.state_dict())
        target_net.eval()
elif algorithm == "Monte-Carlo":
    from utils.memory_mc import ReplayMemoryMonteCarlo
    memory = ReplayMemoryMonteCarlo(10000)
    net = get_model(args)
    net = net.to(device)
    net.eval()

if random_alloy:
    alloy = AlloyGenerator()

if DOUBLE_Q == False:
    # Choose optimizer
    # optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(net.parameters())

criterion = torch.nn.MSELoss()


def select_action(state, greedy=False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            def perturb_state(state, a):
                trial_state = state.copy()
                trial_pos = trial_state.get_positions()
                trial_pos[agent_atom, :] += env.action_space[a]
                trial_state.set_positions(trial_pos)
                return trial_state

            perturbed_states = [perturb_state(state, a) for a in range(n_actions)];
            graph_states = [transformer(sa, agent_atom, n_surf, env.predict_goal_location()-sa.get_positions()[agent_atom], 
                            env.predict_start_location()-sa.get_positions()[agent_atom]) for sa in perturbed_states]
            batch_host = data.collate_atomsdata(graph_states)
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }

            # return net(batch).max(0)[1].view(1, 1)
            #probs = F.softmax(net(batch), dim=0).cpu().detach().numpy().squeeze()

            values = net(batch).cpu().detach().numpy()

            if greedy:
                selected_action = np.argmax(values)
            else:
                scaler = StandardScaler()
                scaled_values = scaler.fit_transform(values).squeeze()
                probs = softmax([scaled_values]).squeeze()
                selected_action = np.random.choice(n_actions, p=probs)

            # Make sure agent doesn't wander outside original unit cell
            new_pos = state.get_positions()[agent_atom]+env.action_space[selected_action]
            if new_pos[0]<=0:
                selected_action = 1 # corresponding to +dx
            elif new_pos[1]<=0:
                selected_action = 3 # corresponding to +dy

            return selected_action, probs[selected_action]
    else:
        return random.randrange(n_actions)



def select_actionQ(state, greedy=False, boltzmann=False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if greedy:
        eps_threshold = 0
        boltzmann = False
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            B = env.predict_goal_location()-state.get_positions()[agent_atom]
            A = env.predict_start_location()-state.get_positions()[agent_atom]
            graph_states = [transformer(state, agent_atom, n_surf, B, A)]
            batch_host = data.collate_atomsdata(graph_states, pin_memory=pin)
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }

            if DOUBLE_Q:
                q_values = 0.5 * (netA(batch) + netB(batch))
            else:
                q_values = net(batch)

            if boltzmann:
                scaler = StandardScaler()
                q_values = q_values.view(-1,1).cpu().detach().numpy()
                #print(q_values)
                #if sum([(q_i - q_values.mean())**2 for q_i in q_values]) == 0.:
                #    print("Identical Q-values!! - Can't do Boltzmann")
                q_values_standard = scaler.fit_transform(q_values)
                q_values_boltzmann = q_values_standard / T_BOLTZ
                probs = softmax([q_values_boltzmann]).squeeze()
                selected_action = np.random.choice(n_actions, p=probs)
            else:
                selected_action = int(q_values.max(1)[1])

    else:
        selected_action = random.randrange(n_actions)

    # Make sure agent doesn't wander outside original unit cell
    new_pos = state.get_positions()[agent_atom]+env.action_space[selected_action]
    if (new_pos[0]<=0) or (new_pos[1]<=0):
        selected_action = 0 # corresponding to B
    elif (new_pos[0]>= env.atom_object.get_cell()[0, 0]) or (new_pos[1]>= env.atom_object.get_cell()[1, 1]):
        selected_action = 0 # corresponding to B

    return selected_action

def optimize_model():
    if len(memory) > BATCH_SIZE:
        transitions = memory.sample(BATCH_SIZE)
        batch = memory.Transition(*zip(*transitions))
        graph_states = [transformer(sa, agent_atom, n_surf, B, A) for (sa, agent_atom, A, B) in zip(batch.state_action, batch.agent_atom, batch.A, batch.B)]
        batch_host = data.collate_atomsdata(graph_states)
        batch_input = {
            k: v.to(device=device, non_blocking=True)
            for (k, v) in batch_host.items()
        }
        batch_target = torch.unsqueeze(torch.cat(batch.ret), 1).float();

        # Reset gradient
        optimizer.zero_grad()
        # Forward, backward and optimize
        outputs = net(batch_input)
        loss = criterion(outputs, batch_target);
        loss.backward()
        optimizer.step()

def optimize_model6():
    if len(memory) > BATCH_SIZE:
        transitions = memory.sample(BATCH_SIZE)
        batch = memory.Transition(*zip(*transitions))

        graph_states = [transformer(s, agent_atom, n_surf, B, A) for (s, agent_atom, A, B) in zip(batch.state, batch.agent_atom, batch.A, batch.B)]
        batch_host = data.collate_atomsdata(graph_states, pin_memory=pin)
        batch_input = {
            k: v.to(device=device, non_blocking=True)
            for (k, v) in batch_host.items()
        }
        batch_target = torch.unsqueeze(torch.cat(batch.ret), 1).float().detach()

        # Forward, backward and optimize
        action_batch = torch.tensor(batch.action, device=device).long().unsqueeze(1)
        state_action_values = net(batch_input).gather(1, action_batch)
        loss = criterion(state_action_values, batch_target)
        optimizer.zero_grad()
        loss.backward()
        # print("loss.grad = " + str(net.state_dict()["readout_mlp.weight"][0,-2:]))
        optimizer.step()

def optimize_modelQ():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = memory.Transition(*zip(*transitions))
    reward_batch = torch.cat(batch.reward)

    # Graph state
    graph_states = [transformer(s, env.agent_number, n_surf, B, A) for (s, A, B) in zip(batch.state, batch.A, batch.B)]
    batch_host = data.collate_atomsdata(graph_states, pin_memory=pin)
    batch_input = {
        k: v.to(device=device, non_blocking=True)
        for (k, v) in batch_host.items()
    }
    # Predicted state-action values
    action_batch = torch.tensor(batch.action, device=device).long().unsqueeze(1)
    if DOUBLE_Q == True:
        # Flip coin
        coin_flip = int(np.random.choice(a=[0,1], size=1))
        if coin_flip == 0:
            state_action_values = netA(batch_input).gather(1, action_batch)
        else:
            state_action_values = netB(batch_input).gather(1, action_batch)
    else:
        state_action_values = net(batch_input).gather(1, action_batch)

    # Graph next_state
    #A_next, B_next = get_A_and_B(batch.next_state, env)
    A_next = tuple([env.predict_start_location() - s.get_positions()[env.agent_number] for s in batch.next_state])
    B_next = tuple([env.predict_goal_location() - s.get_positions()[env.agent_number] for s in batch.next_state])
    graph_states_next = [transformer(s, env.agent_number, n_surf, B, A) for (s, A, B) in zip(batch.next_state, A_next, B_next)]
    batch_host_next = data.collate_atomsdata(graph_states_next, pin_memory=pin)
    batch_input_next = {
        k: v.to(device=device, non_blocking=True)
        for (k, v) in batch_host_next.items()
    }
    # Evalute next state value (regression target)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    if DOUBLE_Q == True:
        if coin_flip == 0:
            a_argmax = netA(batch_input_next).max(1)[1].detach().unsqueeze(1)
            next_state_values[non_final_mask] = netB(batch_input_next).detach().gather(1, a_argmax).squeeze()
        else:
            a_argmax = netB(batch_input_next).max(1)[1].detach().unsqueeze(1)
            next_state_values[non_final_mask] = netA(batch_input_next).detach().gather(1, a_argmax).squeeze()
    else:
        next_state_values[non_final_mask] = target_net(batch_input_next).max(1)[0].detach()

    state_action_values_target = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, state_action_values_target.float().unsqueeze(1))
    loss = criterion(state_action_values, state_action_values_target.float().unsqueeze(1))

    # Optimize the model
    if DOUBLE_Q == True:
        if coin_flip == 0:
            optimizerA.zero_grad
            loss.backward()
            for param in netA.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            optimizerA.step()
        else:
            optimizerB.zero_grad
            loss.backward()
            for param in netB.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            optimizerB.step()
    else:
        optimizer.zero_grad()
        loss.backward()
        #for param in net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        optimizer.step()

def test_trained_agent(env):
    if random_alloy:
        # Sample top two layers
        alloy_atoms = alloy.sample()
        slab, slab_b = alloy.get_relaxed_alloy_map(alloy_atoms=alloy_atoms, map=4)

        # Initialize reinforcement learning environment
        env = ASE_RL_Env(
            initial_state=slab.copy(),
            goal_state=slab_b.copy(),
            agent_number=agent_atom,
            view=False,
            view_force=False
        )

    for i in range(num_episodes_test):
        print("Target episode: " + str(i) + "/" + str(num_episodes_test))
        if (i > 0) and (i % 5 == 0):
            greedy = True
        else:
            greedy = False
        # Initialize the environment and state
        env.reset()
        state = env.atom_object.copy()

        reward_total = 0
        states = []
        actions = []
        rewards = []
        A_vec = []
        B_vec = []
        for t in count():
            #print(np.min(env.pos))
            #if min(env.pos)<0:
            #    print("NEGATIVE POSITIONS!")
 
            states.append(state)
            agent_pos = env.pos[env.agent_number]
            A_vec.append(env.predict_start_location() - agent_pos)
            B_vec.append(env.predict_goal_location() - agent_pos)
            
            # Select and perform an action
            action = select_actionQ(state, greedy = greedy, boltzmann=boltzmann)
            state_action, next_state, reward, done, info = env.step(action)

            # Update accumulated reward for current episode
            reward_total += reward
            reward = torch.tensor([reward], device=device)

            if algorithm == "Q-learning":
                # Store the transition in memory
                memory.push(state, action, next_state, reward, A_vec[-1], B_vec[-1])
                if t % Q_update == 0:
                    optimize_modelQ()

            # Move to the next state
            state = next_state.copy()

            # Save new observation to list
            actions.append(action)
            rewards.append(reward)

            if done:
                
                break

        # Save episode data
        summary.save_episode_RL(env, reward_total, info, states, net, t)
        if algorithm == "Monte-Carlo":
            # Calculate return for all visited states in the episode
            G = 0
            for it in np.arange(t,-1, -1):
                G = GAMMA * G + rewards[it]
                memory.push(states[it], actions[it], G, env.agent_number, A_vec[it], B_vec[it])
                if dublicate_mirror:
                    #mirror_state = states[it].copy()
                    #mirror_state.set_positions(mirror(mirror_state.get_positions(), B_vec[it], n_surf))
                    #mirror_action = mirror_action_space[actions[it]]
                    #memory.push(mirror_state, mirror_action, G, env.agent_number, A_vec[it], B_vec[it])
                    # Swap
                    mirror_state = states[it].copy()
                    mirror_state_pos = mirror_state.get_positions()
                    mirror_state_pos[:, [1, 0]] = mirror_state_pos[:, [0, 1]]
                    mirror_state.set_positions(mirror_state_pos)
                    A_vec[it][[0, 1]] = A_vec[it][[1, 0]]
                    B_vec[it][[0, 1]] = B_vec[it][[1, 0]]
                    mirror_action = mirror_action_space[actions[it]]
                    memory.push(mirror_state, mirror_action, G, env.agent_number, A_vec[it], B_vec[it])
            # Optimize model (no longer after every episode)
            if (i > 0) and (i % 1 == 0):
                optimize_model6()
        elif algorithm == "Q-learning":
            # Update the target network, copying all weights and biases in DQN
            if DOUBLE_Q == False:
                if i % TARGET_UPDATE == 0:
                    target_net.load_state_dict(net.state_dict())

    # Update data and plot
    summary._update_data_RL()
    summary.save_plot()

#####################################################################
############################# Main loop #############################
#####################################################################

summary = PerformanceSummary(
    env,
    net,
    output_dir,
    num_episodes_train,
    num_episodes_test,
    off_policy=True,
    policy1="Random Agent",
    policy2="RL Agent"
)

steps_done = 0
for i_episode in range(num_episodes):
    print("Policy1 episode: " + str(i_episode) + "/" + str(num_episodes))

    # Initialize the environment and state
    env.reset()
    state = env.atom_object

    states = []
    actions = []
    rewards = []
    reward_total = 0
    prob_bs = []

    for t in count():

        states.append(state)
        
        # Select and perform an action using RandomAgent
        agent_pos = env.pos[env.agent_number]
        agent_to_start = env.predict_start_location() - agent_pos
        agent_to_goal = env.predict_goal_location() - agent_pos
        action, prob_b = agent.select_action(agent_to_start, agent_to_goal, t, env.max_iter)

        # Select and perform an action
        state_action, next_state, reward, done, info = env.step(action)
        
        # Update accumulated reward for current episode
        reward_total += reward
        reward = torch.cuda.FloatTensor([reward], device=device)

        # Move to the next state
        state = next_state

        # Save new observation to list
        actions.append(action)
        rewards.append(reward)
        # prob_bs.append(prob_b)

        if done:
            # Save episode data
            summary.save_episode_behavior(env=env, total_reward=reward_total, info=info, states=states)

            # Calculate return for all visited states in the episode
            #G = 0
            #for i in np.arange(t,-1, -1):
            #    G = GAMMA * G + rewards[i]
            #    memory_mc.push(state_actions[i], G, env.agent_number, agent_to_start, agent_to_goal)
            
            # Train deep RL agent
            #optimize_model()

            # Test trained agent
            if (i_episode > 0) and (i_episode % num_episodes_train == 0):
                summary._update_data_behavior()
                test_trained_agent(env)

            break
    # Update the target network, copying all weights and biases in DQN
    #if i_episode % TARGET_UPDATE == 0:
    #    target_net.load_state_dict(policy_net.state_dict())

#summary.save_plot()
