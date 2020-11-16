
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import softmax

import math
import numpy as np
import random
from itertools import count
import logging
import os

# Import surf-rider functionality
from envs.ASE_rl_env import ASE_RL_Env
from models.random_agent import RandomAgent
from utils.memory_mc import ReplayMemoryMonteCarlo
from utils.slab_params import *

import schnet_edge_model
import data



# Set up matplotlib
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize reinforcement learning environment
env = ASE_RL_Env(
    initial_state=slab.copy(),
    goal_state=slab_b.copy(),
    hollow_neighbors=hollow_neighbors,
    goal_dists=dist_B,
    goal_dists_periodic=dist_B_periodic,
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
output_dir = os.path.join(script_dir, 'runs/model_output/')
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
    net = schnet_edge_model.SchnetModel(
        num_interactions=args.num_interactions,
        hidden_state_size=args.node_size,
        cutoff=args.cutoff,
        update_edges=args.update_edges,
        normalize_atomwise=args.atomwise_normalization,
        **kwargs
    )
    return net

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0
EPS_END = 0
EPS_DECAY = 1000

num_episodes = 100
num_episodes_train = 10
num_episodes_test  = 5

num_interactions = 3
node_size = 64
cutoff = 4.0 # assert that cutoff>np.cos(np.pi/6)*env.atom_object.get_cell().lengths()[0]
update_edges = True
atomwise_normalization = True
max_steps = 2000
learning_rate = 0.0001

class args_wrapper():
    def __init__(self, num_interactions: int, node_size: int, cutoff: float, update_edges: bool, atomwise_normalization: bool, 
                 max_steps: int, device: torch.device, learning_rate: float, output_dir):
        self.num_interactions = num_interactions
        self.node_size = node_size
        self.cutoff = cutoff
        self.update_edges = update_edges
        self.atomwise_normalization = atomwise_normalization
        self.max_steps = max_steps
        self.device = device
        self.learning_rate = learning_rate
        self.output_dir = output_dir

args=args_wrapper(num_interactions, node_size, cutoff, update_edges, atomwise_normalization, max_steps, device, learning_rate)
memory_mc = ReplayMemoryMonteCarlo(1000)
transformer = data.TransformAtomsObjectToGraph(cutoff=args.cutoff)

# Initialise model
net = get_model(args)
net = net.to(device)
net.eval()

# Choose optimizer
#optimizer = optim.RMSprop(net.parameters())
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
criterion = torch.nn.MSELoss()


def select_action(state):
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
                trial_pos[agent_atom, :] += env.action_space[a][0]
                trial_state.set_positions(trial_pos)
                return trial_state

            perturbed_states = [perturb_state(state, a) for a in range(n_actions)]
            graph_states = [transformer(sa) for sa in perturbed_states]
            batch_host = data.collate_atomsdata(graph_states)
            batch = {
                k: v.to(device=device, non_blocking=True)
                for (k, v) in batch_host.items()
            }

            # return net(batch).max(0)[1].view(1, 1)
            #probs = F.softmax(net(batch), dim=0).cpu().detach().numpy().squeeze()

            values = net(batch).cpu().detach().numpy()
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(values).squeeze()

            probs = softmax([scaled_values]).squeeze()
            selected_action = np.random.choice(n_actions, p=probs)

            return selected_action, probs[selected_action]
    else:
        return random.randrange(n_actions)


def optimize():
    transitions = memory_mc.sample(BATCH_SIZE)
    batch = memory_mc.Transition(*zip(*transitions))
    graph_states = [transformer(sa) for sa in batch.state_action]
    batch_host = data.collate_atomsdata(graph_states)
    batch_input = {
        k: v.to(device=device, non_blocking=True)
        for (k, v) in batch_host.items()
    }
    batch_target = torch.unsqueeze(torch.cat(batch.ret), 1)

    # Reset gradient
    optimizer.zero_grad()
    # Forward, backward and optimize
    outputs = net(batch_input)
    loss = criterion(outputs, batch_target)
    loss.backward()
    optimizer.step()


def plot_reward():
    plt.figure(2)
    plt.clf()
    reward_t = torch.tensor(total_rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    #if is_ipython:
    #    display.clear_output(wait=True)
    #    display.display(plt.gcf())


def plot_reward_and_energy_barrier():
    
    # Styling
    fig_width = 3.3
    fig_height = 2.1

    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 6})

    colors = [
        '#1f77b4',  # muted blue
        '#d62728',  # brick red
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf',  # blue-teal
    ]

    color_iter = iter(colors)

    # Plot total rewards
    color = next(color_iter)
    ax.plot(
        np.arange(len(RL_total_rewards_avg)),
        RL_total_rewards_avg,
        zorder=1,
        label="Avg. total reward",
        color=color,
    )
    ax.fill_between(
        x=np.arange(len(RL_total_rewards_avg)),
        y1=RL_total_rewards_avg - RL_total_rewards_std,
        y2=RL_total_rewards_avg + RL_total_rewards_std,
        alpha=0.5,
        zorder=1,
        color=color,
    )

    # Plot energy barriers
    color = next(color_iter)
    ax.plot(
        np.arange(len(RL_total_rewards_avg)),
        RL_energy_barriers_avg,
        zorder=2,
        label="Avg. energy barrier",
        color=color,
    )
    ax.fill_between(
        x=np.arange(len(RL_energy_barriers_avg)),
        y1=RL_energy_barriers_avg - RL_energy_barriers_std,
        y2=RL_energy_barriers_avg + RL_energy_barriers_std,
        alpha=0.5,
        zorder=2,
        color=color,
    )

    fig.savefig(os.path.join(output_dir, 'reward_and_energy_barrier.pdf'))


def test_trained_agent():

    global RL_total_rewards
    global RL_energy_barriers
    global RL_energy_profiles
    global best_barrier
    global best_images

    global RL_total_rewards_avg
    global RL_total_rewards_std
    global RL_energy_barriers_avg
    global RL_energy_barriers_std

    for i in range(num_episodes_test):

        # Initialize the environment and state
        env.reset()
        state = env.atom_object
        reward_total = 0
        states = []
        states.append(state)

        for t in count():

            # Select and perform an action
            action, prob = select_action(state)
            state_action, next_state, reward, done, info = env.step(action)

            # Update accumulated reward for current episode
            reward_total += reward

            # Move to the next state
            state = next_state
            states.append(state)

            if done:
                RL_total_rewards.append(reward_total)
                RL_energy_barriers.append(env.energy_barrier)
                RL_energy_profiles.append(env.energy_profile)

                # Save data on new best trajectory 
                if env.energy_barrier < best_barrier:
                    best_barrier = env.energy_barrier
                    best_profile = env.energy_profile
                    best_images = states
  
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": i_episode,
                            "best_barrier": best_barrier,
                            "best_profile": best_profile,
                            "best_images": best_images,
                        },
                        os.path.join(args.output_dir, "best_model.pth"),
                    )

                break
    
    RL_total_rewards_avg.append(np.mean(RL_total_rewards[-num_episodes_test:]))
    RL_total_rewards_std.append(np.std(RL_total_rewards[-num_episodes_test:]))
        
    RL_energy_barriers_avg.append(np.mean(RL_energy_barriers[-num_episodes_test:]))
    RL_energy_barriers_std.append(np.std(RL_energy_barriers[-num_episodes_test:]))

    plot_reward_and_energy_barrier()


#####################################################################
############################# Main loop #############################
#####################################################################


total_rewards = []
energy_barriers=[]
energy_profiles=[]

RL_total_rewards = []
RL_energy_barriers = []
RL_energy_profiles = []
RL_total_rewards_avg = []
RL_energy_barriers_avg = []

best_barrier = np.inf
best_profile = []
best_images = []

steps_done = 0
for i_episode in range(num_episodes):
  
    # Initialize the environment and state
    env.reset()

    state=env.atom_object
    reward_total = 0

    state_actions=[]
    rewards = []
    prob_bs = []

    for t in count():
        
        # Select and perform an action using RandomAgent
        agent_pos = env.pos[env.agent_number]
        agent_to_start = env.predict_start_location() - agent_pos
        agent_to_goal = env.predict_goal_location() - agent_pos
        action, prob_b = agent.select_action(agent_to_start, agent_to_goal, t, env.max_iter)

        # Select and perform an action
        action, prob = select_action(state)
        state_action, next_state, reward, done, info = env.step(action)
        
        
        # Update accumulated reward for current episode
        reward_total += reward
        reward = torch.tensor([reward], device=device)

        # Move to the next state
        state = next_state

        # Save new observation to list
        state_actions.append(state_action)
        rewards.append(reward)
        prob_bs.append(prob_b)

        if done:
            total_rewards.append(reward_total)
            energy_barriers.append(env.energy_barrier)
            energy_profiles.append(env.energy_profile)

            # Calculate return for all visited states in the episode
            G = 0
            for i in np.arange(t,-1, -1):
                G = GAMMA * G + rewards[i]
                memory_mc.push(state_actions[i], G)
            
            # plot_reward()

            optimize_model()

            # Test trained model
            if i_episode % num_episodes_train == 0:
                test_trained_agent()

            break
    # Update the target network, copying all weights and biases in DQN
    #if i_episode % TARGET_UPDATE == 0:
    #    target_net.load_state_dict(policy_net.state_dict())