
from typing import List
import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.optimize import GPMin
from ase.constraints import FixAtoms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import os

class ASE_RL_Env():

    def __init__(self, initial_state: Atoms, action_space: np.ndarray, 
            goal_state: np.ndarray, hollow_neighbors: List,
            goal_dists: np.ndarray, goal_dists_periodic: np.ndarray,
            agent_number: int,
            goal_th: float=0.02, max_force: float=0.05,
            max_barrier: float=1.5, step_size: float=0.1,
            max_iter: int=50, active_dist: float=4.0,
            view: bool=False, view_force: bool=False):

        self.goal_th = goal_th
        self.max_iter = max_iter
        self.max_force = max_force
        self.max_barrier = max_barrier
        self.view = view
        self.view_force = view_force

        self.atom_object = initial_state.copy()
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.agent_number = agent_number

        calc = EMT() #calc = LennardJones()
        self.atom_object.set_calculator(calc)
        self.energy = self.atom_object.get_potential_energy()
        self.min_energy = self.energy
        self.relaxer = BFGS(self.atom_object)

        self.step_size = step_size
        self.action_space = action_space

        self.hollow_neighbors = hollow_neighbors
        self.goal_dists = goal_dists
        self.goal_dists_periodic=goal_dists_periodic

        self.num_atoms = len(self.initial_state)
        self.active_dist = active_dist

        self.initial_dists = self.atom_object.get_distances(
            a=agent_number,
            indices=hollow_neighbors,
            mic=False
        )

        # Calculate vector from neighbors to agent atom in start state
        start_pos_agent = initial_state.get_positions()[agent_number]
        start_pos_neighbors = initial_state.get_positions()[hollow_neighbors]
        self.agent_neigh_disp_start = start_pos_agent - start_pos_neighbors

        # Calculate vector from neighbors to agent atom in goal state
        goal_pos_agent = goal_state.get_positions()[agent_number]
        goal_pos_neighbors = goal_state.get_positions()[hollow_neighbors]
        self.agent_neigh_disp_goal = goal_pos_agent - goal_pos_neighbors

        self.iter = 0

        # Plotting
        if self.view:
            self.initialize_viewer()


    def initialize_viewer(self):
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.view_init(20, -100)
        self.ax.set_facecolor('white')
        self.ax.grid(color='black', linestyle='-', linewidth=0.1)
        self.tags = self.atom_object.get_tags()
        self.atom_colors = np.array(np.repeat('', self.num_atoms), dtype='object')
        self.atom_colors[self.tags == 0] = 'yellow'         # agent atom
        self.atom_colors[self.tags == 1] = 'sienna'         # layer1
        self.atom_colors[self.tags == 2] = 'darkgoldenrod'  # layer2
        self.atom_colors[self.tags == 3] = 'peru'           # layer3
        self.atom_colors[self.tags == 4] = 'darkgoldenrod'  # layer4
        self.pos = self.atom_object.get_positions()
        pos = self.pos
        for i in range(self.num_atoms):
            self.ax.scatter(
                pos[i, 0], pos[i,1], pos[i, 2],
                zdir='z', s=8000, c=self.atom_colors[i],
                depthshade=False, edgecolors='black'
            )
        self.ax.set_xlabel('x', fontsize=20)
        self.ax.set_ylabel('y', fontsize=20)
        self.ax.set_zlabel('z', fontsize=20)
        self.ax.tick_params(labelsize=15)
        self.ax.set_title('Iteration ' + str(self.iter), fontsize=20)
        plt.show(block=False)

        
        self.script_dir = os.path.dirname(__file__)
        self.results_dir = os.path.join(self.script_dir, 'plots/')
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
    

    def reset(self):
        """
            Resets the environment (back to initial state)
        """
        self.atom_object = self.initial_state.copy()
        self.energy = self.atom_object.get_potential_energy()
        self.min_energy = self.energy
    
        if self.view:
            self.initialize_viewer()


    def step(self, action: np.ndarray):
        """
            Agent takes action and all other atoms are relaxed
        """
        self.iter += 1

        # Action step
        self._take_action(action)

        # Update state
        new_state = self._transition()
        
        # Reward
        reward = self._get_reward()

        # Has episode terminated?
        done, done_info = self._episode_terminated()

        # Future versions will probably contain much
        # more information than just termination info
        info = {}
        info["termination"] = done_info 

        return new_state, reward, done, info


    def _take_action(self, action):
        """
            Updates the positions of the agent atom
            according to the chosen action
        """
        # Allow agent atom to be moved
        if self.iter > 1:
            self.mask = np.repeat(False, self.num_atoms)
            constraint = FixAtoms(mask=self.mask)
            self.atom_object.set_constraint(constraint)

        # Increment position by agent action
        self.pos = self.atom_object.get_positions()
        self.pos[self.agent_number, :] += action
        self.atom_object.set_positions(self.pos)

        return None


    def _transition(self):
        """
            Relaxes neighboring atoms in response to the recent action
        """
        all_dists = self.atom_object.get_distances(
            a=self.agent_number,
            indices=list(range(self.num_atoms)),
            mic=True
        )
        self.mask = all_dists > self.active_dist

        if self.iter/self.max_iter > 0.5:
            self.active_dist += 0.1
            #self.max_force *= 0.99

        self.mask[self.agent_number] = True
        constraint = FixAtoms(mask=self.mask)
        self.atom_object.set_constraint(constraint)
        self.relaxer = BFGS(self.atom_object)
        #self.relaxer = GPMin(self.atom_object)
        self.relaxer.run(fmax=self.max_force)
        # print(self.relaxer.)
        self.pos = self.atom_object.get_positions()

        return self.pos


    def _get_reward(self):
        """
            Outputs reward of transition
        """
        old_energy = self.energy
        self.energy = self.atom_object.get_potential_energy()
        reward = self.energy - old_energy

        # Update minimum energy along path
        self.min_energy = min(self.min_energy, self.energy)

        return reward


    def _step_vector(self, action):
        a = action
        dx = self.step_size

        return None


    def _episode_terminated(self):
        """
            Checks if either 
            a) new structure is equal to goal state, or
            b) energy has increased too much
            c) max interations has been reached
        """

        done = False
        info = "game on"

        # a) Has goal site been reached?
        new_dists = self.atom_object.get_distances(
            a=self.agent_number,
            indices=self.hollow_neighbors,
            mic=False
        )

        if np.linalg.norm(new_dists - self.goal_dists) < self.goal_th:
            new_dists_periodic = self.atom_object.get_distances(
                a=self.agent_number,
                indices=self.hollow_neighbors,
                mic=True
            )
            if np.linalg.norm(new_dists_periodic - self.goal_dists_periodic) < self.goal_th:
                done = True
                info = "Goal"

        # b) Has energy wall been struck?
        if self.energy > (self.min_energy + self.max_barrier):
            done = True
            info = "Wall"

        # c) Has max iterations been reached? .. to be removed
        if self.iter >= self.max_iter:
            done = True
            info = "Max_iter"
    
        return done, info


    def render(self):
        """
            Updates the plot with the new atomic coordinates
        """
        if self.view:

            pos = self.atom_object.get_positions()
            plt.ioff()
            self.ax.clear()

            if self.view_force:
                forces = self.atom_object.get_forces()
                for i in range(self.num_atoms):
                    self.ax.scatter(
                        pos[i, 0], pos[i,1], pos[i, 2],
                        zdir='z', s=8000, c=forces,
                        depthshade=False, edgecolors='black'
                    )
            else:
                for i in range(self.num_atoms):
                    self.ax.scatter(
                        pos[i, 0], pos[i,1], pos[i, 2],
                        zdir='z', s=8000, c=self.atom_colors[i],
                        depthshade=False, edgecolors='black'
                    )
            self.ax.set_title('Iteration ' + str(self.iter), fontsize=20)
            plt.ion()
            plt.draw()
            plt.pause(0.001)

            plt.savefig(
                self.results_dir + 'scatter_iter_' + str(self.iter) + '.png',
                bbox_inches='tight'
            )
        else:
            print("Trying to render but view=False")
        
        return None


    def predict_goal_location(self):
        """
            Calculates the location of the goal state
            (this method could be made simpler by simply 
            inputting the Cartesian coordinate of the agent goal
            since the masked atoms prevent the structure from
            wnadering through the unit cell anyway)
        """
        # Hollow neighbor positions
        hnp = self.atom_object.get_positions()[self.hollow_neighbors]
        goal_prediction = np.mean(hnp + self.agent_neigh_disp_goal, axis=0)

        return goal_prediction


    def predict_start_location(self):
        """
            Calculates the location of the start state
        """
        # Hollow neighbor positions
        hnp = self.atom_object.get_positions()[self.hollow_neighbors]
        start_prediction = np.mean(hnp + self.agent_neigh_disp_start, axis=0)

        return start_prediction