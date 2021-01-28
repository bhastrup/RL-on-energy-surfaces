
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Dict
from ase import Atoms
from envs.ASE_rl_env import ASE_RL_Env
from  schnet_edge_model import SchnetModel


class PerformanceSummary():
    def __init__(self, env: ASE_RL_Env, output_dir: str, num_episodes_train: int, num_episodes_test: int):
        
        self.output_dir = output_dir
        self.n_actions = env.n_actions
        
        self.num_episodes_train = num_episodes_train
        self.num_episodes_test = num_episodes_test
        self.episodes_count = 0
        self.action_counts = np.zeros(self.n_actions)

        self.max_dist = np.linalg.norm(env.predict_start_location() - env.predict_goal_location())

        # Data for behavior policy
        self.best_barrier = np.inf
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
        self.RL_best_barrier = np.inf
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

    def save_episode_behavior(self, env: ASE_RL_Env, total_reward: float, states: List[Atoms]) -> None:

        self.episodes_count += 1

        self.total_rewards.append(total_reward)
        self.energy_barriers.append(env.energy_barrier)
        self.distance_covered.append(np.linalg.norm(env.pos[env.agent_number]-env.predict_start_location()))
        self.distance_goal.append(np.linalg.norm(env.predict_goal_location()-env.pos[env.agent_number]))

        if env.energy_barrier < self.best_barrier:
            self.best_barrier = env.energy_barrier
            self.best_profile = env.energy_profile
            [s.set_calculator(None) for s in states]
            self.best_images = states

        return None


    def save_episode_RL(self, env: ASE_RL_Env, total_reward: float, info: str, states: List[Atoms],
        net: SchnetModel, optimizer: torch.optim.Adam) -> None:

        self.RL_total_rewards.append(total_reward)
        self.RL_energy_barriers.append(env.energy_barrier)
        self.RL_distance_covered.append(np.linalg.norm(env.pos[env.agent_number]-env.predict_start_location()))
        self.RL_distance_goal.append(np.linalg.norm(env.predict_goal_location()-env.pos[env.agent_number]))
        self.RL_info.append(info)

        # Save data on new best trajectory 
        if env.energy_barrier < self.RL_best_barrier:
            self.RL_best_barrier = env.energy_barrier
            self.RL_best_profile = env.energy_profile
            [s.set_calculator(None) for s in states]
            self.RL_best_images = states

            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": self.episodes_count * self.num_episodes_train,
                    "best_barrier": self.RL_best_barrier,
                    "best_profile": self.RL_best_profile,
                    "best_images": self.RL_best_images,
                },
                os.path.join(self.output_dir, "best_model.pth"),
            )

        return None

    def _update_data_behavior(self) -> None:
        
        # Append mean and std values performance arrays
        self.total_rewards_avg.append(np.mean(self.total_rewards[-self.num_episodes_train:]))
        self.total_rewards_std.append(np.std(self.total_rewards[-self.num_episodes_train:]))

        self.energy_barriers_avg.append(np.mean(self.energy_barriers[-self.num_episodes_train:]))
        self.energy_barriers_std.append(np.std(self.energy_barriers[-self.num_episodes_train:]))

        self.distance_covered_avg.append(np.mean(self.distance_covered[-self.num_episodes_train:]))
        self.distance_covered_std.append(np.std(self.distance_covered[-self.num_episodes_train:]))

        self.distance_goal_avg.append(np.mean(self.distance_goal[-self.num_episodes_train:]))
        self.distance_goal_std.append(np.std(self.distance_goal[-self.num_episodes_train:]))

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

        return None


    def save_plot(self) -> None:
    
        # Styling
        fig_width = 3.1 
        fig_height = 3.7

        plt.style.use('ggplot')
        plt.rcParams.update({'font.size': 3})

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

        step_range = np.arange(len(self.RL_total_rewards_avg)) * self.num_episodes_train

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height), constrained_layout=True)

        #################################################################
        ##################### Plot total rewards ########################
        #################################################################

        ax[0, 0].plot(
            step_range,
            self.total_rewards_avg,
            zorder=2,
            label="Behavior policy",
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
        ax[0, 0].plot(
            step_range,
            self.RL_total_rewards_avg,
            zorder=2,
            label="Target policy",
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
        ax[1, 0].plot(
            step_range,
            self.energy_barriers_avg,
            zorder=2,
            label="Behavior policy",
            color=behavior_color,
        )
        ax[1, 0].fill_between(
            x=step_range,
            y1=np.array(self.energy_barriers_avg) - np.array(self.energy_barriers_std),
            y2=np.array(self.energy_barriers_avg) + np.array(self.energy_barriers_std),
            alpha=0.5,
            zorder=1,
            color=behavior_color,
        )
        ax[1, 0].plot(
            step_range,
            self.RL_energy_barriers_avg,
            zorder=2,
            label="Target policy",
            color=target_color,
        )
        ax[1, 0].fill_between(
            x=step_range,
            y1=np.array(self.RL_energy_barriers_avg) - np.array(self.RL_energy_barriers_std),
            y2=np.array(self.RL_energy_barriers_avg) + np.array(self.RL_energy_barriers_std),
            alpha=0.5,
            zorder=1,
            color=target_color,
        )
        ax[1, 0].set(xlabel='Number of training episodes')
        ax[1, 0].set(ylabel='Avg. energy barrier')
        ax[1, 0].legend(loc='upper right')

        #################################################################
        ###################   Plot distance covered/to goal   ###########
        #################################################################

        # Behavior 
        ax[0, 1].plot(
            step_range,
            self.distance_covered_avg,
            zorder=2,
            label="Distance to start - behavior",
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
            label="Distance to goal - behavior",
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
            label="Distance to start",
            color=target_color,
        )
        ax[0, 1].fill_between(
            x=step_range,
            y1=np.array(self.RL_distance_covered_avg) - np.array(self.RL_distance_covered_std),
            y2=np.array(self.RL_distance_covered_avg) + np.array(self.RL_distance_covered_std),
            alpha=0.5,
            zorder=1,
            color=target_color,
        )
        color = next(color_iter)
        ax[0, 1].plot(
            step_range,
            self.RL_distance_goal_avg,
            zorder=2,
            label="Distance to goal",
            color=color,
        )
        ax[0, 1].fill_between(
            x=step_range,
            y1=np.array(self.RL_distance_goal_avg) - np.array(self.RL_distance_goal_std),
            y2=np.array(self.RL_distance_goal_avg) + np.array(self.RL_distance_goal_std),
            alpha=0.5,
            zorder=1,
            color=color,
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

        print("self.RL_info"); print(self.RL_info)
        print("info_list"); print(info_list)
        print("t_count"); print(t_count)
        print("self.RL_info_count"); print(self.RL_info_count)

        ax[1, 1].stackplot(step_range, self.RL_info_count, labels=termination_types)
        ax[1, 1].legend(loc='upper right')
        ax[1, 1].set(xlabel='Number of training episodes')

        fig.savefig(os.path.join(self.output_dir, 'PerformanceSummary.pdf'))

        return None
