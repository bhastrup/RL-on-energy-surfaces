
from typing import List
import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.optimize import BFGS
from ase.constraints import FixAtoms

class AtomicEnv():

    def __init__(self, atom_object: Atoms, action_space: np.ndarray, 
            goal_state: np.ndarray, hollow_neighbors: List,
            goal_dists: np.ndarray, goal_dists_periodic: np.ndarray,
            agent_number: int,
            goal_th: float=0.02, max_force: float=0.05,
            max_barrier: float=1.5, step_size: float=0.1,
            max_iter: int=50, active_dist: float=4.0):

        self.atom_object = atom_object
        self.agent_number = agent_number
        calc = EMT()
        #calc = LennardJones()
        self.atom_object.set_calculator(calc)
        self.energy = self.atom_object.get_potential_energy()
        self.min_energy = self.energy
        self.max_force = max_force
        self.max_barrier = max_barrier
        self.relaxer = BFGS(self.atom_object)
        self.step_size = step_size
        self.action_space = action_space
        self.reward_accum = 0
        self.goal_state = goal_state
        self.hollow_neighbors = hollow_neighbors
        self.goal_dists = goal_dists
        self.goal_dists_periodic=goal_dists_periodic
        self.goal_th = goal_th
        self.max_iter = max_iter
        self.num_atoms = atom_object.get_global_number_of_atoms()
        self.active_dist = active_dist

        self.initial_dists = self.atom_object.get_distances(
            a=agent_number,
            indices=hollow_neighbors,
            mic=False
        )

        # Calculate vector from neighbors to agent atom in start state
        start_pos_agent = atom_object.get_positions()[agent_number]
        start_pos_neighbors = atom_object.get_positions()[hollow_neighbors]
        self.agent_neigh_disp_start = start_pos_agent - start_pos_neighbors

        # Calculate vector from neighbors to agent atom in goal state
        goal_pos_agent = goal_state.get_positions()[agent_number]
        goal_pos_neighbors = goal_state.get_positions()[hollow_neighbors]
        self.agent_neigh_disp_goal = goal_pos_agent - goal_pos_neighbors

        self.iter = 0


    def step(self, action: int):
        """
            Agent takes action and all other atoms are relaxed 
        """
        self.iter += 1

        # Action step
        old_pos = self.atom_object.get_positions()
        new_pos = old_pos

        if self.iter > 1:
            mask = np.repeat(False, self.num_atoms)
            constraint = FixAtoms(mask=mask)
            self.atom_object.set_constraint(constraint)

        new_pos[self.agent_number, :] += action
        self.atom_object.set_positions(new_pos)

        # Update state
        all_dists = self.atom_object.get_distances(self.agent_number, indices=list(range(self.num_atoms)))
        mask = all_dists > self.active_dist

        if self.iter/self.max_iter > 0.5:
            self.active_dist += 0.1
            #self.max_force *= 0.99

        mask[self.agent_number] = True
        constraint = FixAtoms(mask=mask)
        self.atom_object.set_constraint(constraint)
        self.relaxer = BFGS(self.atom_object)
        self.relaxer.run(fmax=self.max_force)
        
        # Reward
        old_energy = self.energy
        self.energy = self.atom_object.get_potential_energy()
        reward = self.energy - old_energy
        self.reward_accum += reward

        # Update minimum energy along path
        self.min_energy = min(self.min_energy, self.energy)

        # Has episode terminated?
        done, info = self._episode_terminated()

        return reward, done, info
    

    def _episode_terminated(self):
        """
            Checks if either 
            a) new structure is equal to goal state, or
            b) energy has increased too much
            c) max interations has been reached
        """

        done = False
        info = "game on"

        # a) has goal site been reached?
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

        # b) has energy wall been struck?
        if self.energy > (self.min_energy + self.max_barrier):
            done = True
            info = "Wall"

        # c) has max iterations been reached?
        if self.iter >= self.max_iter:
            done = True
            info = "Max_iter"
    
        return done, info

    def _get_reward(self):
        """
            Outputs reward of action
        """

        return None

    def _take_action(self, action):

        old_pos = self.atom_object.get_positions()
        
        return None


    def _step_vector(self, action):
        a = action
        dx = self.step_size

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