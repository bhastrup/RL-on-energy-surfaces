import math
import random
from itertools import count
import pickle
from itertools import islice

# Import surf-rider functionality
from envs.ASE_rl_env import ASE_RL_Env
from models.random_agent import RandomAgent
from models.DQN_network import DQN
from utils.memory import ReplayMemory
from utils.slab_params import *
from utils.voxel_utils import get_3d_grid, get_voxel_repr


def save_memory_to_pickle(data, pickle_file="memory.p"):
    # Save transitions into a pickle file.
    pickle.dump(data, open( pickle_file, "wb" ) )
    return None


def load_memory_from_pickle(pickle_file="memory.p"):
    # # Load the transitions back from the pickle file.
    memory = pickle.load( open( pickle_file, "rb" ) )
    return memory


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

traj_dict = {}
num_episodes = 5000
episode_reward = []

n_episode_save = 100

for i_episode in range(num_episodes):

    # Initialize the environment and state
    env.reset()
    reward_total = 0
    one_episode_dict = {}

    for t in count():

        one_trans_dict = {}
        one_trans_dict['state'] = env.atom_object.get_positions()
        one_trans_dict['energy_before'] = env.atom_object.get_potential_energy()
        

        # Select and perform an action using RandomAgent   
        agent_pos = env.atom_object.get_positions()[env.agent_number]
        agent_to_start = env.predict_start_location() - agent_pos
        agent_to_goal = env.predict_goal_location() - agent_pos
        # Choose action
        action = agent.select_action(agent_to_start, agent_to_goal, t, env.max_iter)
        _, reward, done, done_info = env.step(action)
        
        one_trans_dict['action'] = action
        one_trans_dict['state_after'] = env.atom_object.get_positions()
        one_trans_dict['energy_after'] = env.atom_object.get_potential_energy()
        
        
        # Update accumulated reward for current episode
        reward_total += reward

        one_episode_dict[str(t)] = one_trans_dict

        if done:
            episode_reward.append(reward_total)

            one_episode_dict['done_info'] = done_info
            traj_dict[str(i_episode)] = one_episode_dict

            # Every now and then
            if i_episode % n_episode_save == 0:
                save_memory_to_pickle(traj_dict, pickle_file="memory.p")

            break



#save_memory_to_pickle(traj_dict, pickle_file="memory.p")
#mem = load_memory_from_pickle(pickle_file="memory.p")

#for ep_id, ep_dict in mem.items():
#    print("\nEpisode ID:", ep_id)
#    for trans_id, trans_dict in islice(ep_dict.items(), 0, len(ep_dict)-1):
#        print("\nTransition ID:", trans_id)
#        print("\nPotential Energy Diff:", trans_dict['energy_after'] - trans_dict['energy_before'])

