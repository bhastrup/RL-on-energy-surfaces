

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import math
import random
from itertools import count
import pickle

# Import surf-rider functionality
from envs.ASE_rl_env import ASE_RL_Env
from models.random_agent import RandomAgent
from models.DQN_network import DQN
from utils.memory import ReplayMemory
from utils.slab_params import *
from utils.voxel_utils import get_3d_grid, get_voxel_repr

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

# Define agent
k = 20 # softmax coefficient
sigma = 1.2 # exploration factor
agent = RandomAgent(action_space=env.action_space, k=k, sigma=sigma)


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
            return policy_net(state.double()).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def plot_reward():
    plt.figure(2)
    #plt.clf()
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
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


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = memory.Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).double()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device).double()
    next_state_values[non_final_mask] = target_net(non_final_next_states.double()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def save_memory_to_pickle(data, pickle_file="memory.p"):
    # Save transitions into a pickle file.
    pickle.dump(data, open( pickle_file, "wb" ) )
    return None


def load_memory_from_pickle(pickle_file="memory.p"):
    # # Load the transitions back from the pickle file.
    memory = pickle.load( open( pickle_file, "rb" ) )
    return memory

#####################################################
##################  Training  #######################
#####################################################

# Deep learning hyper parameters
BATCH_SIZE = 8
GAMMA = 1 # 0.999
EPS_START = 0.7
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 4

# Voxel representation parameters
radius = 4
sigma = 0.5
unit_cell_len = 2*radius + 1
n_grid = 28
grid_3d = get_3d_grid(unit_cell_len, n_grid)

# Define DQN networks
n_actions = env.n_actions # (consider just 6 action, i.e. +-x, +-y and +-z)
policy_net = DQN(n_grid, n_grid, n_grid, n_actions).to(device)
policy_net = policy_net.double()
policy_net.eval()

target_net = DQN(n_grid, n_grid, n_grid, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net = target_net.double()
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(500)

traj_dict = {}
num_episodes = 1
episode_reward = []
steps_done = 0
# plt.ion()
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = get_voxel_repr(env.atom_object, env.agent_number, env.predict_goal_location(),
                           grid_3d, n_grid, radius=radius, sigma=sigma)
    reward_total = 0
    for t in count():

        # Select and perform an action
        #action = select_action(state)
        #_, reward, done, done_info = env.step(action.item())
        
        # Random agent
        agent_pos = env.atom_object.get_positions()[env.agent_number]
        agent_to_start = env.predict_start_location() - agent_pos
        agent_to_goal = env.predict_goal_location() - agent_pos
        # Choose action
        action = agent.select_action(agent_to_start, agent_to_goal, t, env.max_iter)
        _, reward, done, done_info = env.step(action)
        #print(env.action_space[action][0])
        #print(env.action_space[action])
        action = torch.tensor([action], device=device, dtype=torch.long)
        # print(action.item())
        

        # Update accumulated reward for current episode
        reward_total += reward
        print("Reward = " + str(reward))
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = get_voxel_repr(env.atom_object, env.agent_number, env.predict_goal_location(),
                                        grid_3d, n_grid, radius=radius, sigma=sigma)
        else:
            next_state = None
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        env.render()
        # Perform one step of the optimization (on the target network)
        # optimize_model()
        if done:
            print(done_info)
            episode_reward.append(reward_total)
            plot_reward()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


print('Complete')
plt.ioff()
plt.show(block=False)

save_memory_to_pickle(data=memory.memory, pickle_file="memory.p")