
import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage.filters import gaussian_filter
from typing import List, Tuple, Dict

from ase import Atoms
from envs.ASE_rl_env import ASE_RL_Env
from schnet_edge_model import SchnetModel
from ase.io import Trajectory
from ase.io import write
from asap3 import EMT
from ase.constraints import FixAtoms
from ase.build import fcc100

def save_data_to_pickle(data, pickle_file="pickle_data.p"):
    # Save transitions into a pickle file.
    pickle.dump(data, open( pickle_file, "wb" ) )
    return None


class PerformanceSummary():
    def __init__(self, env: ASE_RL_Env, net: SchnetModel, output_dir: str, num_episodes_train: int, num_episodes_test: int, off_policy: bool=False, policy1: str="Off policy", policy2: str="RL Agent"):
        
        self.output_dir = output_dir
        self.n_actions = env.n_actions
        
        self.num_episodes_train = num_episodes_train
        self.num_episodes_test = num_episodes_test
        self.off_policy = off_policy

        self.policy1 = policy1
        self.policy2 = policy2

        self.action_counts = np.zeros(self.n_actions)

        self.max_dist = np.linalg.norm(env.predict_start_location() - env.predict_goal_location())

        # Whenever a new best path is found it is saved as traj file indexed by self.best_count
        self.best_count = -1

        # Data for behavior policy
        self.episodes_count = 0
        self.best_barrier = env.max_barrier
        self.best_barrier_vec = []
        self.best_profile = []
        self.best_images = []

        self.total_rewards = []
        self.total_rewards_avg = []
        self.total_rewards_std = []

        self.energy_barriers = []
        self.energy_barriers_avg = []
        self.energy_barriers_std = []

        self.distance_covered = []
        self.distance_covered_avg = []
        self.distance_covered_std = []

        self.distance_goal = []
        self.distance_goal_avg = []
        self.distance_goal_std = []

        # Data for target policy
        self.RL_episodes_count = 0
        self.RL_step_count = 0
        self.RL_best_barrier = env.max_barrier
        self.RL_best_barrier_vec = []
        self.RL_best_profile = []
        self.RL_best_images = []

        self.RL_total_rewards = []
        self.RL_total_rewards_avg = []
        self.RL_total_rewards_std = []

        self.RL_energy_barriers = []
        self.RL_energy_barriers_avg = []
        self.RL_energy_barriers_std = []

        self.RL_distance_covered = []
        self.RL_distance_covered_avg = []
        self.RL_distance_covered_std = []

        self.RL_distance_goal = []
        self.RL_distance_goal_avg = []
        self.RL_distance_goal_std = []

        self.RL_info = []
        self.RL_info_count = [[],[],[]]

        self.RL_readout_mlp_w = [] # [[]] * model["model"]["readout_mlp.weight"].size()[1]
        self.RL_readout_mlp_w_avg = []
        #self.readout_mlp_w = np.empty_like(model["model"]["readout_mlp.weight"])
        #self.RL_readout_mlp_c_half_w = []
        #self.RL_readout_mlp_c_half_w_avg = []


        # Make criteria switch for new best model
        self.start_goal_dist = np.linalg.norm(env.predict_goal_location()-env.pos[env.agent_number])
        self.best_dist_to_goal = 10*self.start_goal_dist
        self.best_criteria = "distance"

        # Heat map
        self.xedges = np.linspace(0, env.atom_object.get_cell()[0,0], 51)
        self.yedges = np.linspace(0, env.atom_object.get_cell()[1,1], 51)
        self.H = np.zeros((len(self.xedges)-1, len(self.yedges)-1))
        self.start_pos = env.initial_state.get_positions()[env.agent_number]
        self.start_pos_top_layer = env.initial_state.get_positions()[[atom.tag < 2 for atom in env.initial_state]]
        self.goal_pos = env.goal_state.get_positions()[env.agent_number]
        self.RL_final_pos = []
        self.best_path_pos = []

    def save_episode_behavior(self, env: ASE_RL_Env, total_reward: float, info: str, states: List[Atoms]) -> None:

        if self.off_policy:
            self.episodes_count += 1

            self.total_rewards.append(total_reward)
            self.energy_barriers.append(env.energy_barrier)
            self.distance_covered.append(np.linalg.norm(env.pos[env.agent_number]-env.predict_start_location()))
            self.distance_goal.append(np.linalg.norm(env.predict_goal_location()-env.pos[env.agent_number]))

            if (info == 'Goal') and (env.energy_barrier < self.best_barrier):
                self.best_barrier = env.energy_barrier
                self.best_profile = env.energy_profile
                [s.set_calculator(None) for s in states]
                self.best_images = states

        return None


    def save_episode_RL(self, env: ASE_RL_Env, total_reward: float, info: str, states: List[Atoms],
        net: SchnetModel, n_steps: int) -> None:

        self.RL_episodes_count += 1
        self.RL_step_count += n_steps
        self.RL_total_rewards.append(total_reward)
        self.RL_energy_barriers.append(env.energy_barrier) # These are not really barriers.

        self.RL_distance_covered.append(np.linalg.norm(env.pos[env.agent_number]-env.predict_start_location()))
        self.RL_distance_goal.append(np.linalg.norm(env.predict_goal_location()-env.pos[env.agent_number]))
        self.RL_info.append(info)

        if net.state_dict()["readout_mlp.weight"].shape[0] == 6:
            self.RL_readout_mlp_w.append(net.state_dict()["readout_mlp.weight"][0].squeeze().tolist())
        else:
            self.RL_readout_mlp_w.append(net.state_dict()["readout_mlp.weight"].squeeze().tolist())
        #self.RL_readout_mlp_c_half_w.append(net.state_dict()["readout_mlp_c_half.0.weight"].squeeze().tolist())

        # Heatmap
        agent_pos_epi = np.array([atoms.get_positions()[env.agent_number] for atoms in states])
        H_new, _, _ = np.histogram2d(agent_pos_epi[:, 0], agent_pos_epi[:, 1], bins=(self.xedges, self.yedges))
        self.H += H_new.T
        self.RL_final_pos.append(env.pos[env.agent_number])

        # Did we observe a new best path in this episode? Initially the criteria is goal distance, later energy barrier
        if self.best_criteria == "distance":
            if self.RL_distance_goal[-1] < self.best_dist_to_goal:
                # We have a new best path
                new_best = True
                self.best_dist_to_goal = self.RL_distance_goal[-1]

                # Switch criteria?
                if self.RL_info.count('Goal') > 0:
                    self.best_criteria = "energy_barrier"
            else:
                new_best = False
        else:
            if (info == 'Goal') and (env.energy_barrier < self.RL_best_barrier):
                new_best = True
                self.RL_best_barrier = env.energy_barrier
                self.RL_best_profile = env.energy_profile
                [s.set_calculator(None) for s in states]
                self.RL_best_images = states
            else:
                new_best = False

        # Save data on new best trajectory
        if new_best:
            # Save states to ASE trajectory file
            self.best_count += 1
            self.RL_best_path_pos = [s.get_positions()[env.agent_number] for s in states]
            #best_traj = Trajectory(os.path.join(self.output_dir, "best_path_" + str(self.best_count) + ".traj"), 'w')
            #for state in states:
            #    best_traj.write(state)

            # Save other relevant data episode data to pickle
            traj_dict = {
                "RL_best_images": self.RL_best_images,
                "RL_best_barrier": self.RL_best_barrier,
                "RL_best_profile": self.RL_best_profile,
                "RL_best_path_pos": self.RL_best_path_pos,
                "best_count": self.best_count,
                "RL_episodes_count": self.RL_episodes_count,
                "RL_total_rewards": self.RL_total_rewards,
                "RL_energy_barriers": self.RL_energy_barriers,
                "RL_distance_covered": self.RL_distance_covered,
                "RL_distance_goal": self.RL_distance_goal,
                "RL_info": self.RL_info,
                "RL_readout_mlp_w": self.RL_readout_mlp_w
                #"RL_readout_mlp_c_half_w": self.RL_readout_mlp_c_half_w
            }
            save_data_to_pickle(traj_dict, pickle_file=os.path.join(self.output_dir, "pickle_data.p"))

            # Save pytorch neural network
            torch.save(
                {
                    "model": net.state_dict(),
                    #"optimizer": optimizer.state_dict(),
                    #"step": self.episodes_count * self.num_episodes_train,
                    "steps": self.RL_step_count,
                    "episodes": self.RL_episodes_count,
                    "best_barrier": self.RL_best_barrier,
                    "best_profile": self.RL_best_profile,
                    "best_images": self.RL_best_images,
                },
                os.path.join(self.output_dir, "best_model.pth"),
            )

        return None

    def _update_data_behavior(self) -> None:
        
        if self.off_policy:
            # Append mean and std values performance arrays
            self.total_rewards_avg.append(np.mean(self.total_rewards[-self.num_episodes_train:]))
            self.total_rewards_std.append(np.std(self.total_rewards[-self.num_episodes_train:]))

            self.energy_barriers_avg.append(np.mean(self.energy_barriers[-self.num_episodes_train:]))
            self.energy_barriers_std.append(np.std(self.energy_barriers[-self.num_episodes_train:]))

            self.distance_covered_avg.append(np.mean(self.distance_covered[-self.num_episodes_train:]))
            self.distance_covered_std.append(np.std(self.distance_covered[-self.num_episodes_train:]))

            self.distance_goal_avg.append(np.mean(self.distance_goal[-self.num_episodes_train:]))
            self.distance_goal_std.append(np.std(self.distance_goal[-self.num_episodes_train:]))

            self.best_barrier_vec.append(self.best_barrier)

        return None


    def _update_data_RL(self) -> None:

        # Append mean and std values performance arrays
        self.RL_total_rewards_avg.append(np.mean(self.RL_total_rewards[-self.num_episodes_test:]))
        self.RL_total_rewards_std.append(np.std(self.RL_total_rewards[-self.num_episodes_test:]))

        self.RL_energy_barriers_avg.append(np.mean(self.RL_energy_barriers[-self.num_episodes_test:]))
        self.RL_energy_barriers_std.append(np.std(self.RL_energy_barriers[-self.num_episodes_test:]))

        self.RL_distance_covered_avg.append(np.mean(self.RL_distance_covered[-self.num_episodes_test:]))
        self.RL_distance_covered_std.append(np.std(self.RL_distance_covered[-self.num_episodes_test:]))
        
        self.RL_distance_goal_avg.append(np.mean(self.RL_distance_goal[-self.num_episodes_test:]))
        self.RL_distance_goal_std.append(np.std(self.RL_distance_goal[-self.num_episodes_test:]))

        self.RL_readout_mlp_w_avg.append(np.mean(self.RL_readout_mlp_w[-self.num_episodes_test:], axis=0))
        #self.RL_readout_mlp_c_half_w_avg.append(np.mean(self.RL_readout_mlp_c_half_w[-self.num_episodes_test:], axis=0))

        self.RL_best_barrier_vec.append(self.RL_best_barrier)
        
        return None


    def save_plot(self) -> None:
    
        # Styling
        fig_width = 3.1 
        fig_height = 3.7

        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 3})
        plt.rcParams.update({'lines.linewidth': 1})


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
        target_color = next(color_iter)
        behavior_color = next(color_iter)
        other_color = next(color_iter)

        step_range = np.arange(len(self.RL_total_rewards_avg)) * self.num_episodes_test #self.num_episodes_train

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(fig_width, fig_height), constrained_layout=True)

        #################################################################
        ##################### Plot total rewards ########################
        #################################################################
        if self.off_policy:
            ax[0, 0].plot(
                step_range,
                self.total_rewards_avg,
                zorder=2,
                label=self.policy1,
                color=behavior_color,
            )
            ax[0, 0].fill_between(
                x=step_range,
                y1=np.array(self.total_rewards_avg) - np.array(self.total_rewards_std),
                y2=np.array(self.total_rewards_avg) + np.array(self.total_rewards_std),
                alpha=0.5,
                zorder=1,
                color=behavior_color,
            )
        # Target
        ax[0, 0].plot(
            step_range,
            self.RL_total_rewards_avg,
            zorder=2,
            label=self.policy2,
            color=target_color,
        )
        ax[0, 0].fill_between(
            x=step_range,
            y1=np.array(self.RL_total_rewards_avg) - np.array(self.RL_total_rewards_std),
            y2=np.array(self.RL_total_rewards_avg) + np.array(self.RL_total_rewards_std),
            alpha=0.5,
            zorder=1,
            color=target_color,
        )
        # plt.setp( ax[0,0].get_xticklabels(), visible=False)
        ax[0, 0].set(ylabel='Avg. total reward')
        ax[0, 0].legend(loc='upper right')

        #################################################################
        ##################### Plot energy barriers ######################
        # ###############################################################
        if self.off_policy:
            ax[1, 0].plot(
                step_range,
                self.best_barrier_vec,
                zorder=2,
                label=self.policy1,
               color=behavior_color,
            )
            #ax[1, 0].fill_between(
            #    x=step_range,
            #    y1=np.array(self.energy_barriers_avg) - np.array(self.energy_barriers_std),
            #    y2=np.array(self.energy_barriers_avg) + np.array(self.energy_barriers_std),
            #    alpha=0.5,
            #    zorder=1,
            #    color=behavior_color,
            #)
        # Target
        ax[1, 0].plot(
            step_range,
            self.RL_best_barrier_vec,
            zorder=2,
            label=self.policy2,
            color=target_color,
        )
        #ax[1, 0].fill_between(
        #    x=step_range,
        #    y1=np.array(self.RL_energy_barriers_avg) - np.array(self.RL_energy_barriers_std),
        #    y2=np.array(self.RL_energy_barriers_avg) + np.array(self.RL_energy_barriers_std),
        #    alpha=0.5,
        #    zorder=1,
        #    color=target_color,
        #)
        ax[1, 0].set(xlabel='Number of training episodes')
        ax[1, 0].set(ylabel='Best energy barrier')
        ax[1, 0].legend(loc='upper right')

        #################################################################
        ###################   Plot distance covered/to goal   ###########
        #################################################################
        if self.off_policy:
            # Behavior 
            ax[0, 1].plot(
                step_range,
                self.distance_covered_avg,
                zorder=2,
                label="Distance to start - " + self.policy1,
                color=behavior_color,
            )
            ax[0, 1].fill_between(
                x=step_range,
                y1=np.array(self.distance_covered_avg) - np.array(self.distance_covered_std),
                y2=np.array(self.distance_covered_avg) + np.array(self.distance_covered_std),
                alpha=0.5,
                zorder=1,
                color=behavior_color,
            )
            color = next(color_iter)
            ax[0, 1].plot(
                step_range,
                self.distance_goal_avg,
                zorder=2,
                label="Distance to goal - " + self.policy1,
                color=color,
            )
            ax[0, 1].fill_between(
                x=step_range,
                y1=np.array(self.distance_goal_avg) - np.array(self.distance_goal_std),
                y2=np.array(self.distance_goal_avg) + np.array(self.distance_goal_std),
                alpha=0.5,
                zorder=1,
                color=color,
            )
        # Target
        ax[0, 1].plot(
            step_range,
            self.RL_distance_covered_avg,
            zorder=2,
            label="Distance to start - " + self.policy2,
            color=target_color,
        )
        ax[0, 1].fill_between(
            x=step_range,
            y1=np.array(self.RL_distance_covered_avg) - np.array(self.RL_distance_covered_std),
            y2=np.array(self.RL_distance_covered_avg) + np.array(self.RL_distance_covered_std),
            alpha=0.5,
            zorder=1,
            color=target_color,
            ec='face'
        )
        color = next(color_iter)
        ax[0, 1].plot(
            step_range,
            self.RL_distance_goal_avg,
            zorder=2,
            label="Distance to goal - " + self.policy2,
            color=color,
        )
        ax[0, 1].fill_between(
            x=step_range,
            y1=np.array(self.RL_distance_goal_avg) - np.array(self.RL_distance_goal_std),
            y2=np.array(self.RL_distance_goal_avg) + np.array(self.RL_distance_goal_std),
            alpha=0.5,
            zorder=1,
            color=color,
            ec='face'
        )
        ax[0, 1].legend(loc='upper right')
        ax[0, 1].set(ylabel='Avg. final distance to start and goal')

        #################################################################
        #####################   Plot termination info      ##############
        #################################################################

        # Plot termination info (https://python-graph-gallery.com/250-basic-stacked-area-chart/)

        info_list = self.RL_info[-self.num_episodes_test:]
        termination_types = ['Goal', 'Wall', 'Max_iter']

        t_count = [info_list.count(t_type) for t_type in termination_types]
        
        for i in range(0, len(termination_types)):
            self.RL_info_count[i].append(t_count[i])

        ax[1, 1].stackplot(step_range, self.RL_info_count, labels=termination_types)
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set(xlabel='Number of training episodes')

        #################################################################
        ##################   Plot readout_mlp weights      ##############
        #################################################################

        weights = np.array(self.RL_readout_mlp_w_avg)
        ax[2, 0].plot(
            step_range,
            weights[:, :-2],
            #zorder=2,
            linewidth=0.4
        )

        ax[2, 0].plot(step_range, weights[:, -2], linewidth=1.5, color="black", label="$r_A$ weight")
        ax[2, 0].plot(step_range, weights[:, -1], linewidth=1.5, color="blue", label="$r_B$ weight")
        ax[2, 0].legend(loc='upper right')
        ax[2, 0].set(xlabel='Number of training episodes')
        ax[2, 0].set(ylabel='Avg. readout_mlp weight')

        #################################################################
        ##############   Plot agent position heat map      ##############
        #################################################################

        ax[2, 1].scatter(self.start_pos_top_layer[:-1, 0], self.start_pos_top_layer[:-1, 1], s=325, c='none', edgecolor='grey', marker='o', alpha=0.75)

        H = gaussian_filter(self.H, sigma=0.15)
        heat = ax[2, 1].imshow(np.log(H+1), interpolation='nearest', origin='lower',
            extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])
        cb = fig.colorbar(heat, ax=ax[2, 1], pad=-0.08) # pad=-0.03 without ggplot
        ax[2, 1].set(xlabel='$x$')
        ax[2, 1].set(ylabel='$y$')

        tick_label_list = [1, 2, 3, 4, 5, 6, 7, 8]
        ax[2, 1].set_yticks(tick_label_list)
        ax[2, 1].set_yticklabels(tick_label_list)
        ax[2, 1].set_xticks(tick_label_list)
        ax[2, 1].set_xticklabels(tick_label_list)
        ax[2, 1].set_xlim([self.xedges[0], self.xedges[-1]])
        ax[2, 1].set_ylim([self.yedges[0], self.yedges[-1]])

        ax[2, 1].grid(None) # False without ggplot
        cb.set_label('number of visits (log-scale)')

        RL_final_pos = np.array(self.RL_final_pos)
        wiggle_x = np.random.normal(loc=0, scale=0.05, size=len(RL_final_pos))
        wiggle_y = np.random.normal(loc=0, scale=0.05, size=len(RL_final_pos))
        ax[2, 1].scatter(RL_final_pos[:, 0] + wiggle_x, RL_final_pos[:, 1] + wiggle_y, s=0.2, c='black', marker='.', linewidths=0)
        ax[2, 1].scatter([self.start_pos[0], self.goal_pos[0]], [self.start_pos[1], self.goal_pos[1]], s=2, c='red', marker='x', alpha=1)

        ax[2, 1].text(self.start_pos[0]-0.4, self.start_pos[1] + 0.4, 'START', c='white', fontsize=2)
        ax[2, 1].text(self.goal_pos[0]-0.4, self.goal_pos[1] + 0.4, 'GOAL', c='white', fontsize=2)

        path = np.array(self.RL_best_path_pos)
        ax[2, 1].plot(path[:, 0], path[:, 1], linewidth=0.2, color="red")


        #################################################################
        ##############   Plot readout_mlp_c_half weights      ###########
        #################################################################

        #weights = np.array(self.RL_readout_mlp_c_half_w_avg)
        #ax[3, 0].plot(
        #    step_range,
        #    weights[:, :-3],
        #    #zorder=2,
        #    linewidth=0.4
        #)

        #ax[3, 0].plot(step_range, weights[:, -3], linewidth=1.5, color="black", label="$\alpha$ weight")
        #ax[3, 0].plot(step_range, weights[:, -2], linewidth=1.5, color="black", label="$\beta$ weight")
        #ax[3, 0].plot(step_range, weights[:, -1], linewidth=1.5, color="blue", label="$r$ weight")
        #ax[3, 0].legend(loc='upper right')
        #ax[3, 0].set(xlabel='Number of training episodes')
        #ax[3, 0].set(ylabel='Avg. readout_mlp_c_half weight')


        fig.savefig(os.path.join(self.output_dir, 'PerformanceSummary_A.pdf'))


        #################################################################
        ##############   Plot energy profile and height     #############
        #################################################################

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height), constrained_layout=True)

        ax[0].plot(self.RL_best_profile, linewidth=0.8, color="black")
        ax[0].set(xlabel='step')
        ax[0].set(ylabel='Energy [eV]')

        ax[1].plot(path[:, 2]-path[0, 2], linewidth=0.8, color="black")
        ax[1].set(xlabel='step')
        ax[1].set(ylabel='Height [Å]')
        ax[1].set_ylim(-1, 2)

        fig.savefig(os.path.join(self.output_dir, 'PerformanceSummary_B.pdf'))

        return None
