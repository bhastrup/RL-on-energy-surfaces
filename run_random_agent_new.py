
from typing import List

import numpy as np
import torch
# import argparse


from pathlib import Path, PurePath
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from envs.ASE_rl_env import ASE_RL_Env
from models.random_agent import RandomAgent
# from utils.slab_params import *
from utils.alloy import *

# Initialize reinforcement learning environment
env = ASE_RL_Env(
    initial_state=slab.copy(),
    goal_state=slab_b.copy(),
    agent_number=agent_atom,
    view=True,
    view_force=False
)



# Get number of actions from the environment class
n_actions = env.n_actions

n_surf = - np.array([0.,0.,1.])


# Random agent
k = 8 #100 # softmax coefficient
sigma = 0.5 # exploration factor
agent = RandomAgent(action_space=env.action_space, k=k, sigma=sigma)


num_episodes = 1
rewards_list = []
steps_count = []
break_info = []


for i in range(num_episodes):

    print("Starting episode " + str(i))

    total_reward = 0
    t = 0
 
    env.reset()
    
    traj = Trajectory("epi_" + str(i) + ".traj", 'w')
    while t < (env.max_iter + 1):
        
        print("t = " + str(t))

        print("Agent position: " + str(env.atom_object.get_positions()[env.agent_number]))

	# Select and perform an action using RandomAgent
        agent_pos = env.pos[env.agent_number]
        agent_to_start = env.predict_start_location() - agent_pos
        agent_to_goal = env.predict_goal_location() - agent_pos
        action, prob_b = agent.select_action(agent_to_start, agent_to_goal, t, env.max_iter)

        # Select and perform an action
        state_action, next_state, reward, done, info = env.step(action)

        env.render()

        # Save state to ASE trajectory file
        traj.write(env.atom_object)


        if done:
            print("Done is True, " + info)
            steps_count.append(t)
            break_info.append(info)
            #rewards_list.append(env.reward_accum)
            
            # Save to .traj file
            #write(
            #    filename="episode_" + str(i) + ".traj",
            #    images=images
            #)

            break

        t += 1


