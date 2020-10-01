
from typing import List

import numpy as np
from sklearn.utils.extmath import softmax
import torch
# import argparse

from ase import Atoms
from ase.io import Trajectory
from ase.io import write
from pathlib import Path, PurePath
import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


from ase.visualize import view # run view(atom_objects[0])
from ase.build import fcc111, add_adsorbate
from ase.calculators.abinit import Abinit
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.constraints import FixAtoms

from envs.ASE_rl_env import ASE_RL_Env
from misc import create_action_space, drift_projection

# Specify action space
action_space = create_action_space(step_size=0.1)

# Build ASE slab
slab = fcc111('Cu', size=(2,3,4), vacuum=10.0)
add_adsorbate(slab, 'Cu', 2.08, 'fcc')
calc = EMT()
#calc = LennardJones()
slab.set_calculator(calc)
#view(slab)

# Specify initial configuration, A
dyn = BFGS(slab, trajectory='slab.traj')
dyn.run(fmax=0.05)


# Specify goal configuration, B
slab_b = slab.copy()
slab_b[-1].x += slab.get_cell()[0, 0] / 2
dyn_B = BFGS(slab, trajectory='slab_B.traj')
dyn_B.run(fmax=0.05)
#view(slab_b)

# Specify agent atom
n_atoms = len(slab)
agent_atom = n_atoms - 1

# Calculated if agent atom is in goal hollow site
hollow_neighbors = [12, 13, 14]
dist_A = slab.get_distances(agent_atom, hollow_neighbors, mic=False)
dist_B= slab_b.get_distances(agent_atom, hollow_neighbors, mic=False)
dist_B_periodic = slab_b.get_distances(agent_atom, hollow_neighbors, mic=True)


num_episodes = 1
rewards_list = []
steps_count = []
break_info = []
k = 100 # softmax coefficient
sigma = 3 # exploration factor

for i in range(num_episodes):

    print("Starting episode " + str(i))

    total_reward = 0
    t = 0
 
    # Initialize/reset reinforcement learning environment
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
    
    traj = Trajectory("epi_" + str(i) + ".traj", 'w')
    while t < (env.max_iter + 1):
        
        print("t = " + str(t))

        # Choose action
        agent_pos = env.atom_object.get_positions()[env.agent_number]
        agent_to_start = env.predict_start_location() - agent_pos
        agent_to_goal = env.predict_goal_location() - agent_pos

        start_proj = drift_projection(env.action_space, agent_to_start)
        goal_proj = drift_projection(env.action_space, agent_to_goal)

        lambda_sm = (env.max_iter-t)/env.max_iter # lambda softmax
        if t>0:
            p_action = softmax([((-lambda_sm)*start_proj/np.linalg.norm(agent_to_start)*sigma
                                + (1-lambda_sm)*goal_proj)*k])[0]
        else:
            p_action = None

        action = env.action_space[np.random.choice(
            len(env.action_space),
            size=1,
            p=p_action
        )][0]

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


