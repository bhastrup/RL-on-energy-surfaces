
from typing import List

import numpy as np
import torch
# import argparse


from pathlib import Path, PurePath
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from slab_params import *
from envs.ASE_rl_env import ASE_RL_Env
from models.random_agent import RandomAgent
from utils.memory import ReplayMemory

# Initialize reinforcement learning environment
env = ASE_RL_Env(
    initial_state=slab.copy(),
    goal_state=slab_b.copy(),
    hollow_neighbors=hollow_neighbors,
    goal_dists=dist_B,
    goal_dists_periodic=dist_B_periodic,
    agent_number=agent_atom,
    view=True,
    view_force=False
)

# Define agent
k = 100 # softmax coefficient
sigma = 3 # exploration factor
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

        # Acquire transition progression data
        agent_pos = env.atom_object.get_positions()[env.agent_number]
        agent_to_start = env.predict_start_location() - agent_pos
        agent_to_goal = env.predict_goal_location() - agent_pos

        # Choose action
        action = agent.select_action(agent_to_start, agent_to_goal, t, env.max_iter)

        # Implement action on environment
        new_state, reward, done, info  = env.step(action)
        env.render()

        # Save state to ASE trajectory file
        traj.write(env.atom_object)

        print("Agent position: " + str(env.atom_object.get_positions()[env.agent_number]))

        if done:
            print("Done is True, " + str(info["termination"]))
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


