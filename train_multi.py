

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax

import math
import numpy as np
import pandas as pd
np.seterr(all='raise')
import random
from itertools import count
import logging
import os
from typing import List, Tuple, Dict

# Import surf-rider functionality
from envs.ASE_rl_env import ASE_RL_Env
from models.random_agent import RandomAgent
from utils.memory import ReplayMemory
from utils.memory_mc import ReplayMemoryMonteCarlo

#from utils.slab_params import *
from utils.alloy import *
from utils.alloymap import AlloyGenerator
from utils.summary_muti import PerformanceSummary
from utils.mirror import mirror
from utils.neb import get_neb_energy

import schnet_edge_model
import schnet_edge_model_action
import schnet_edge_model_action_no_cat
import data

from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d_%m_%H_%M")

# Set up matplotlib
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

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
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, 'runs/multi/model_output_' + dt_string + '/')
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
random_alloy = True
algorithm = "Monte-Carlo"
DOUBLE_Q = False
boltzmann = True
dublicate_mirror = False
mirror_action_space = np.array([0, 1, 3, 2, 4, 5])
T_BOLTZ = 0.25
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.1
EPS_END = 0.05
EPS_DECAY = 15000
TARGET_UPDATE = 10
Q_update = 3

E_neb_thres = 1.1

num_episodes = 250000
num_episodes_train = 10
num_episodes_test  = 250

num_interactions = 3
node_size = 64
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
transformer = data.TransformAtomsObjectToGraph(cutoff=args.cutoff)

# Initialise model
if algorithm == "Q-learning":
    from utils.memory import ReplayMemory
    memory = ReplayMemory(50000)
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
    memory = ReplayMemoryMonteCarlo(50000)
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



def select_actionQ(state, boltzmann=False):
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


def test_trained_agent(env):
    if random_alloy:
        # Sample top two layers
        alloy_atoms = alloy.sample()
        slab, slab_b = alloy.get_relaxed_alloy_map(alloy_atoms=alloy_atoms, map=4)
        slab_up = alloy.get_relaxed_single(alloy_atoms, 'up')
        slab_right = alloy.get_relaxed_single(alloy_atoms, 'right')

        # Calculate NEB energy
        E_up1, _, n_Ag_up1, height_up1  = get_neb_energy(slab, slab_up)
        E_up2, _, n_Ag_up2, height_up2 = get_neb_energy(slab_up, slab_b)
        E_right1, _, n_Ag_right1, height_right1 = get_neb_energy(slab, slab_right)
        E_right2, _, n_Ag_right2, height_right2 = get_neb_energy(slab_right, slab_b)

        E_neb = min(
            max(E_up1, E_up2),
            max(E_right1, E_right2)
        )

        brige_atoms_E = {n_Ag_up1: E_up1,
                         n_Ag_up2: E_up2,
                         n_Ag_right1: E_right1,
                         n_Ag_right2: E_right2}

        data = {'n_Ag': [n_Ag_up1, n_Ag_up2, n_Ag_right1, n_Ag_right2],
                'E_neb': [E_up1, E_up2, E_right1, E_right2],
                'height': [height_up1, height_up2, height_right1, height_right2]}

        df = pd.DataFrame(data)
        summary.save_multi(df)
        T_neb = num_episodes_test

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

            states.append(state)
            agent_pos = env.pos[env.agent_number]
            A_vec.append(env.predict_start_location() - agent_pos)
            B_vec.append(env.predict_goal_location() - agent_pos)
            
            # Select and perform an action
            action = select_actionQ(state, boltzmann=boltzmann)
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
                    mirror_state = states[it].copy()
                    mirror_state.set_positions(mirror(mirror_state.get_positions(), B_vec[it], n_surf))
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

        if (info=='Goal') and (env.energy_barrier<E_neb_thres*E_neb) and (T_neb==num_episodes_test):
            T_neb = i
            break

    # Update data and plot
    summary._update_data_RL()
    summary.save_plot()
    summary.save_map_data(T_neb, num_episodes_test, E_neb_thres)

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
