
import sys, os
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from typing import List, Tuple, Dict
import numpy as np
from sklearn.utils.extmath import softmax

from ase import Atoms
#from ase.calculators.emt import EMT
from asap3 import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from .bfgs_max import BFGS as BFGS_MAX
from ase.optimize import GPMin
from ase.constraints import FixAtoms

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation



class ASE_RL_Env():

    def __init__(self, initial_state: Atoms, goal_state: Atoms,
        agent_number: int, view: bool=False, view_force: bool=False):

        self.goal_th = 0.6
        self.max_iter = 100
        self.max_force = 0.05
        self.max_barrier = 1.0
        self.step_size = 0.15
        self.active_dist = 4.5
        self.max_optim_steps = 10 # steps before fmax begins to increase by 10% in BFGS_MAX
        self.constant_reward = 0.01 # -0.003
        self.progression_reward = 0.5
        self.final_barrier_reward = 10
        self.goal_reward = abs(1.5*self.max_barrier*self.final_barrier_reward)
        self.profile_coef = 4

        self.view = view
        self.view_force = view_force
        self.initial_state = initial_state
        self.goal_state = goal_state
        calc = EMT()
        self.goal_state.set_calculator(calc)
        self.goal_energy = self.goal_state.get_potential_energy()
        self.num_atoms = len(self.initial_state)
        self.agent_number = agent_number

        self.reset()

        self.n_actions = len(self.action_space)
        # Distance from goal agent to nearest neighbor atoms
        self.goal_neighbors = (np.linalg.norm(
            goal_state.get_positions()[agent_number] - goal_state.get_positions()[:-1], axis=1)).argsort()[:4]

        self.dist_to_goal_neighbors = goal_state.get_positions()[agent_number] - goal_state.get_positions()[self.goal_neighbors]

        # Think everything dist related below is now redundant 
        #self.initial_dists = self.atom_object.get_distances(
        #    a=agent_number,
        #    indices=hollow_neighbors,
        #    mic=False
        #)

        # Calculate vector from neighbors to agent atom in start state
        #start_pos_agent = initial_state.get_positions()[agent_number]
        #start_pos_neighbors = initial_state.get_positions()[hollow_neighbors]
        #self.agent_neigh_disp_start = start_pos_agent - start_pos_neighbors

        # Calculate vector from neighbors to agent atom in goal state
        #goal_pos_agent = goal_state.get_positions()[agent_number]
        #goal_pos_neighbors = goal_state.get_positions()[hollow_neighbors]
        #self.agent_neigh_disp_goal = goal_pos_agent - goal_pos_neighbors


        self.script_dir = os.path.dirname(__file__)

        # Plotting
        # if self.view:
        #    self.initialize_viewer()

        

        # Create directory to save trajectory files for hessian?? noope
        #self.results_dir = os.path.join(self.script_dir, 'plots/')
        #if not os.path.isdir(self.results_dir):
        #    os.makedirs(self.results_dir)


    def get_action_space(self) -> np.ndarray:
        """
            Creates flattened array of action displacement vectors in 3d
            Should probably be a list instead
        """
        action_space = np.zeros((3,3,3), object)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    action_space[i,j,k] = np.array([i-1,j-1,k-1])

        action_space = action_space.flatten()
        if max(action_space[13]) == 0:
            action_space = np.delete(action_space, 13)

        for i in range(len(action_space)):
            action_space[i] = np.divide(action_space[i], np.linalg.norm(action_space[i])) * self.step_size

        # np.stack(action_space)
        return action_space


    def get_action_space_6(self) -> np.ndarray:
        """
            Creates flattened array of action displacement vectors in 3d
            Should probably be a list instead
        """
        action_space = np.zeros(6, object)

        action_space[0] = np.array([-1, 0, 0])
        action_space[1] = np.array([1, 0, 0])
        action_space[2] = np.array([0, -1, 0])
        action_space[3] = np.array([0, 1, 0])
        action_space[4] = np.array([0, 0, -1])
        action_space[5] = np.array([0, 0, 1])

        for i in range(len(action_space)):
                action_space[i] = np.divide(action_space[i], np.linalg.norm(action_space[i])) * self.step_size
        
        return action_space


    def update_action_space_internal(self) -> np.ndarray:
        """
            Creates flattened array of action displacement vectors in 3d
            Should probably be a list instead
        """

        B = self.predict_goal_location()-self.pos[self.agent_number, :]
        n = np.array([0., 0., -1.])
        nB = np.cross(n, B)
        BnB = np.cross(B, nB)

        action_space = np.zeros(6, object)

        action_space[0] = B
        action_space[1] = -B
        action_space[2] = nB
        action_space[3] = -nB
        action_space[4] = BnB
        action_space[5] = -BnB

        for i in range(len(action_space)):
                action_space[i] = np.divide(action_space[i], np.linalg.norm(action_space[i])) * self.step_size

        self.action_space = action_space

        return None


    def initialize_viewer(self) -> None:
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

        # Specify where to save the plots
        self.results_dir = os.path.join(self.script_dir, 'plots/')
        if not os.path.isdir(self.results_dir):
            os.makedirs(self.results_dir)
    

    def reset(self):
        """
            Resets the environment (back to initial state)
        """
        self.iter = 0
        self.atom_object = self.initial_state.copy()
        calc = EMT()
        self.atom_object.set_calculator(calc)
        self.relaxer = BFGS(self.atom_object)
        self.energy = self.atom_object.get_potential_energy()
        self.min_energy = self.energy
        self.energy_profile = [self.energy]
        self.energy_barrier = 0
        self.pos = self.atom_object.get_positions()
        self.goal_dist = self.dist_to_goal()
        self.start_dist = self.dist_to_start()
        self.update_action_space_internal()

        if self.view:
            self.initialize_viewer()


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
            Agent takes action and all other atoms are relaxed
        """
        self.iter += 1

        # Action step
        state_action = self._take_action(action)

        # Update state
        new_state = self._transition()
        
        # Reward
        reward = self._get_reward()

        # Has episode terminated?
        done, done_info, terminal_reward = self._episode_terminated()
        
        reward += terminal_reward

        # Future versions will probably contain much
        # more information than just termination info
        # info = {}
        # info["termination"] = done_info

        return state_action, new_state, reward, done, done_info


    def _take_action(self, action: int) -> Atoms:
        """
            Updates the positions of the agent atom
            according to the chosen action
        """

        # Increment position by agent action
        self.pos = self.atom_object.get_positions()
        # print(self.action_space[action])
        self.pos[self.agent_number, :] += self.action_space[action]
        if (self.pos[self.agent_number, 0] < 0) or (self.pos[self.agent_number, 1] < 0):
            print("Agent moves to neighbor unit cell!! - new pos=" + str(self.pos[self.agent_number, :]))

        self.atom_object.set_positions(self.pos)
        self.update_action_space_internal()

        return self.atom_object.copy()


    def _transition(self) -> Atoms:
        """
            Relaxes neighboring atoms in response to the recent action
        """
        all_dists = self.atom_object.get_distances(
            a=self.agent_number,
            indices=list(range(self.num_atoms)),
            mic=True
        )
        self.mask = all_dists > self.active_dist

        #if self.iter/self.max_iter > 0.5:
            # self.active_dist += 0.05
            #self.max_force *= 0.99

        self.mask[self.agent_number] = True
        constraint = FixAtoms(mask=self.mask)
        self.atom_object.set_constraint(constraint)
        self.relaxer = BFGS_MAX(self.atom_object)
        #self.relaxer = BFGS(self.atom_object)
        #self.relaxer = GPMin(self.atom_object)
        self.relaxer.run(fmax=self.max_force, steps=4*self.max_optim_steps)
        self.pos = self.atom_object.get_positions()

        # Finally, remove the constraint from all atoms
        # (necessary for .get_forces())
        self.mask = np.repeat(False, self.num_atoms)
        constraint = FixAtoms(mask=self.mask)
        self.atom_object.set_constraint(constraint)

        return self.atom_object.copy()


    def _get_reward(self) -> float:
        """
            Outputs reward of transition
        """
        # Energy
        old_energy = self.energy
        self.energy = self.atom_object.get_potential_energy()
        
        # Distance
        old_goal_dist = self.goal_dist
        old_start_dist = self.start_dist

        self.goal_dist = self.dist_to_goal()
        self.start_dist = self.dist_to_start()

        reward = -0.25*(self.energy - old_energy) \
            + self.constant_reward \
            + self.progression_reward * (self.start_dist - old_start_dist) \
            - self.progression_reward * (self.goal_dist - old_goal_dist)

        # Update energy profile, minimum energy and barrier along path
        self.energy_profile.append(self.energy)
        self.min_energy = min(self.min_energy, self.energy)
        self.energy_barrier = max(self.energy_barrier, self.energy-self.min_energy)

        return reward


    def _episode_terminated(self) -> Tuple[bool, str, float]:
        """
            Checks if either 
            a) new structure is equal to goal state, or
            b) energy has increased too much
            c) max interations has been reached
        """

        done = False
        info = "game on"
        terminal_reward = 0

        
        if self.test_goal(self.goal_th):
            # a) Has goal site been reached?
            done = True
            info = "Goal"
            terminal_reward = - self.final_barrier_reward*self.energy_barrier + self.goal_reward \
                              + (self.energy - self.goal_energy) \
                              - self.profile_coef * np.mean(np.array(self.energy_profile) - self.energy_profile[0])
        elif self.energy_barrier > self.max_barrier:
            # b) Has energy wall been struck?
            done = True
            info = "Wall"
            terminal_reward = -self.max_barrier
        elif self.iter >= self.max_iter:
            # c) Has max iterations been reached? 
            done = True
            info = "Max_iter"
            if self.test_goal(3*self.goal_th):
                terminal_reward = -self.energy_barrier*(1+min(1, self.dist_to_goal()/(2*self.goal_th)))

        return done, info, terminal_reward


    def test_goal(self, goal_th: float) -> bool:
        """
            Tests if goal has been reached to precision goal_th
        """

        goal = False
        
        #current_dists = self.atom_object.get_distances(
        #    a=self.agent_number,
        #    indices=self.goal_neighbors,
        #    mic=False
        #)

        # Compare current distance vector to goal neighbors to distance vectors in goal state
        #if np.linalg.norm(current_dists - self.dist_to_goal_neighbors) < goal_th:
        #    goal = True

        # Now we are just using the naive absolute goal criteria. 
        # Won't work for more interesting systems with flexible goal locations.    
        if self.dist_to_goal() < goal_th:
            goal = True

        return goal


    def dist_to_start(self) -> float:

        absolute_start_dist = np.linalg.norm(
            self.pos[self.agent_number] - self.initial_state.get_positions()[self.agent_number]
        )

        return absolute_start_dist

    def dist_to_goal(self) -> float:

        absolute_goal_dist = np.linalg.norm(
            self.pos[self.agent_number] - self.goal_state.get_positions()[self.agent_number]
        )

        return absolute_goal_dist


    def render(self) -> None:
        """
            Updates the plot with the new atomic coordinates
            WARNING: Very slow. Do not use for actual data collection/training)
        """
        if self.view:

            pos = self.atom_object.get_positions()
            plt.ioff()
            self.ax.clear()

            if self.view_force:
                # Should forces be shown after action instead of relaxation?
                forces = self.atom_object.get_forces()
                forces_magnitude = np.linalg.norm(forces, axis=1)
                for i in range(self.num_atoms):
                    #print('force ' + str(i) + ' = ' + str(forces_magnitude[i]))
                    alpha = (1/self.max_force)*forces_magnitude[i]
                    alpha = min(1, max(alpha, 0))
                    self.ax.scatter(
                        pos[i, 0], pos[i,1], pos[i, 2],
                        zdir='z', s=8000, c=self.atom_colors[i],
                        alpha=alpha,
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


    def predict_goal_location(self) -> np.ndarray:
        """
            Calculates the location of the goal state
            (this method could be made simpler by simply 
            inputting the Cartesian coordinate of the agent goal
            since the masked atoms prevent the structure from
            wnadering through the unit cell anyway)
        """
        # Hollow neighbor positions
        # hnp = self.atom_object.get_positions()[self.hollow_neighbors]
        # goal_prediction = np.mean(hnp + self.agent_neigh_disp_goal, axis=0)

        # Function has been changed since we can ignore the risk of pushing the slab
        # around in the unit cell when bottom layers are fixed anyways
        goal_prediction = self.goal_state.get_positions()[self.agent_number]

        return goal_prediction


    def predict_start_location(self) -> np.ndarray:
        """
            Calculates the location of the start state
        """
        # Hollow neighbor positions
        # hnp = self.atom_object.get_positions()[self.hollow_neighbors]
        # start_prediction = np.mean(hnp + self.agent_neigh_disp_start, axis=0)

        start_prediction = self.initial_state.get_positions()[self.agent_number]

        return start_prediction
