

# Import surf-rider functionality
from envs.ASE_rl_env import ASE_RL_Env
from models.random_agent import RandomAgent
from utils.slab_params import *
import matplotlib.pyplot as plt
import numpy as np

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

n_steps = 125
k = np.zeros(n_steps)
s = np.zeros(n_steps)
for i in range(n_steps):
    k[i] = agent.k
    s[i] = agent.s
    agent._evolve_params()

#plt.plot(k)
plt.plot(s)
