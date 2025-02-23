import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
import scienceplots
from collections import defaultdict
import os
import yaml
import pandas as pd


def smooth_fct(data, kernel_size=5):
    "Smooth data with convolution of kernel_size"
    # kernel_size=5
    kernel = np.ones(kernel_size) / kernel_size
    convolved_data = np.convolve(data.squeeze(), kernel, mode='same')
    return convolved_data


plt.rcParams.update(bundles.iclr2024(
    family="serif", rel_width=0.9, nrows=1.0, ncols=1.0))
np.random.seed(190)  # for bootstrapping

ourblue = (0.368, 0.507, 0.71)
ourorange = (0.881, 0.611, 0.142)
ourgreen = (0.56, 0.692, 0.195)
ourred = (0.923, 0.386, 0.209)
ourviolet = (0.528, 0.471, 0.701)
ourbrown = (0.772, 0.432, 0.102)
ourlightblue = (0.364, 0.619, 0.782)
ourdarkgreen = (0.572, 0.586, 0.0)
ourdarkred = (0.923, 0.386, 0.)


COLOR_DICT = {'ours': ourorange,
              'baseline': ourblue,
              }

domain_tasks = {
    "cheetah": ['walk', 'walk_backward', 'run', 'run_backward'],
    "quadruped": ['stand', 'walk', 'run', 'jump'],
    "walker": ['stand', 'walk', 'run', 'flip'],
    "maze": ['room1', 'room2', 'room3', 'room4'],
    "hopper": ['hop', 'stand', 'hop_backward', 'flip', 'flip_backward']
}

BASE_PATH = '/home/nuria/phd/controllable_agent/results_clus'
# dir_figs = '/home/nuria/phd/controllable_agent/figs/exp1'


# group_key = (uncertainty, mix_ratio, add_trunk, update_z_every, sampling) if env != 'maze' else (uncertainty, mix_ratio, add_trunk)
# final_hyperparams = {'hopper': [[(True, 0.3, None, 100, True), ourorange], [(True, 0.3, None, 100, False), ourgreen], [(False, 0.3, None, 100, False), ourblue]],
#                      'maze': [[(True, 0.3, True), ourorange], [(False, 0.3, True), ourblue]],
#                      'cheetah': [[(True, 0.3, None, None, None), ourorange], [(False, 0.3, None, None, None), ourblue]],
#                      'quadruped': [[(True, 0.3, None, None, None), ourorange], [(False, 0.3, None, None, None), ourblue]],
#                      'walker': [[(True, 0.3, None, None, None), ourorange], [(False, 0.3, None, None, None), ourblue]]
#                      }


# paths = [f'{BASE_PATH}/online_fb_quadruped_alltasks_vel',
#          f'{BASE_PATH}/online_fb_quadruped_alltasks',
#          f'{BASE_PATH}/online_fb_cheetah_alltasks',
#          f'{BASE_PATH}/online_fb_maze_alltasks',
#          f'{BASE_PATH}/online_fb_walker_alltasks',
#          f'{BASE_PATH}/online_fb_hopper_alltasks_2',
#          f'{BASE_PATH}/online_fb_hopper_alltasks_baseline',
#          ]

# TASK_PATH = [paths[5], paths[6]]  # hopper:
# TASK_PATH = [paths[2]] : cheetah

####

paths = [f'{BASE_PATH}/fb_quadruped',
         f'{BASE_PATH}/fb_quadruped_2',
         f'{BASE_PATH}/fb_cheetah',
         f'{BASE_PATH}/fb_maze',
         f'{BASE_PATH}/fb_walker',
         f'{BASE_PATH}/fb_hopper',
         ]
# group_key = (uncertainty, mix_ratio, add_trunk, update_z_every, sampling)
final_hyperparams = {'hopper': [[(True, 0.3, None, 100, True), ourorange], [(True, 0.3, None, 100, False), ourgreen], [(False, 0.3, None, 100, False), ourblue]],
                     'maze': [[(True, 0.3, True, 100, True), ourorange], [(True, 0.3, True, 100, False), ourgreen], [(False, 0.3, True, 100, False), ourblue]],
                     'cheetah': [[(True, 0.3, None, 100, True), ourorange], [(True, 0.3, None, 100, False), ourgreen], [(False, 0.3, None, 100, False), ourblue]],
                     'quadruped': [[(True, 0.3, None, 100, True), ourorange], [(True, 0.3, None, 100, False), ourgreen], [(False, 0.3, None, 100, False), ourblue]],
                     'walker': [[(True, 0.3, None, 100, True), ourorange], [(True, 0.3, None, 100, False), ourgreen], [(False, 0.3, None, 100, False), ourblue]]
                     }
dir_figs = '/home/nuria/phd/controllable_agent/figs/exp2'

# TASK_PATH = [paths[5]] #hopper
# TASK_PATH = [paths[2]] #cheetah
# TASK_PATH = [paths[3]] #maze
# TASK_PATH = [paths[4]] #walker
TASK_PATH = [paths[0]]  # quadruped

###

grouped_files = defaultdict(list)


env = [e for e in list(domain_tasks.keys()) if e in TASK_PATH[0]][0]
ignore_files = ['commit.txt', 'job_spec.sh']
files = [os.path.join(t, file) for t in TASK_PATH for file in os.listdir(
    t) if file not in ignore_files]

for exp in files:
    config_path = os.path.join(exp, "config.yaml")
    eval_path = os.path.join(exp, "eval.csv")

    # Ensure both config.yaml and eval.csv exist
    if not os.path.exists(config_path) or not os.path.exists(eval_path):
        print(f'Config path {config_path} or {eval_path} does not exist')
        continue

    # Read config.yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract hyperparameters
    uncertainty = config.get("uncertainty")  # Default to False if missing
    mix_ratio = config.get("agent").get(
        "mix_ratio", None)  # Default to None if missing
    add_trunk = config.get("agent").get(
        "add_trunk", None)  # Default to None if missing
    update_z_every = config.get("agent").get(
        "update_z_every_step", None)  # Default to None if missing
    sampling = config.get("agent").get(
        "sampling", None)  # Default to None if missing
    # Use (uncertainty, mix_ratio) as the group key
    group_key = (uncertainty, mix_ratio, add_trunk, update_z_every,
                 sampling)  # if env != 'maze' else (uncertainty, mix_ratio, add_trunk)
    print(group_key)
    num_eval_frames = config.get("eval_every_frames")
    # Store the eval.csv file path in the corresponding group
    # Dict with keys the different set of params, and values the list of files with same params
    grouped_files[group_key].append(eval_path)

# Dict of dicts. Keys are task names, values are dictionary of group name and value the sequence of rewards
grouped_data = defaultdict(dict)
for task in domain_tasks[env]:
    key_rew = f"episode_reward_{env}_{task}" if env != 'maze' else f"reward_{task}"
    grouped_data[key_rew] = defaultdict(dict)

    for group_key, paths in grouped_files.items():
        grouped_data[key_rew][group_key] = list()
        for path in paths:
            print(f"Group {group_key}, {key_rew}: {path}")
            df = pd.read_csv(path)
            rewards = df[key_rew].tolist()  # Convert column to list
            # Store data in the grouped dictionary
            grouped_data[key_rew][group_key].append(rewards)

key_rew = 'episode_reward'
# Adding avg reward among all tasks
for group_key, paths in grouped_files.items():
    grouped_data[key_rew][group_key] = list()
    for path in paths:
        print(f"Group {group_key}, {key_rew}: {path}")
        df = pd.read_csv(path)
        rewards = df[key_rew].tolist()  # Convert column to list
        # Store data in the grouped dictionary
        grouped_data[key_rew][group_key].append(rewards)


##########################################
# # Plot reward per task
SAVE = True
with plt.style.context(["grid"]):
    fig, axs = plt.subplots(1, len(grouped_data.keys()),
                            figsize=(10, 3))  # 1 plot for each env_task
    plt_num = -1
    for env_task, groups in grouped_data.items():
        plt_num += 1
        max_ylim = 0

        for group_key in groups.keys():
            print(group_key)

            if group_key in list(map(lambda x: x[0], final_hyperparams[env])):
                color = [l[1]
                         for l in final_hyperparams[env] if l[0] == group_key][0]
                # Compute mean and std of the rewards
                rews_seeds = groups[group_key]
                print(
                    f'Number of files for group: {group_key}: {len(rews_seeds)}')
                # In case some exps are longer than others
                min_len = min([len(rew) for rew in rews_seeds])
                rews_seeds = [rew[:min_len] for rew in rews_seeds]

                rewards = np.array(rews_seeds)
                mean = np.mean(rewards, axis=0)
                # mean = smooth_fct(mean, kernel_size=2)
                std = np.std(rewards, axis=0)
                steps = np.arange(len(mean))+1  # add 1 because we start at 1!
                # Show only part of the curve
                if env != 'maze':
                    steps = steps[0:10]
                mean = mean[0:len(steps)]
                std = std[0:len(steps)]

                # print(f"Group {group_key}: {mean[-1]:.2f} ± {std[-1]:.2f}")
                # Plot the mean and std
                label_ = f"ours: {group_key}" if group_key[0] == True else f"baseline: {group_key}"
                axs[plt_num].plot(
                    steps, mean, label=f"{label_}", color=color, linewidth=2.0)
                axs[plt_num].fill_between(
                    steps, mean - std, mean + std, color=color, alpha=0.15)
                title = (' ').join(env_task.split('_')[-2::])
                axs[plt_num].set_title(title, fontsize=20)
                axs[plt_num].set_xlabel(
                    f'Datasize$\\times$ {num_eval_frames}', fontsize=15)
                axs[plt_num].set_ylabel('Task reward', fontsize=15)
                max_ylim = max(max(mean), max_ylim) + 50
                if env == 'hopper':
                    axs[plt_num].set_ylim([0, max_ylim])
                else:
                    axs[plt_num].set_ylim([0, 1000])
                # axs[plt_num].set_xlim([0, 10])
                axs[plt_num].tick_params(axis='x', labelsize=12)
                axs[plt_num].tick_params(axis='y', labelsize=12)
                # Bold and bigger x-axis ticks
                axs[plt_num].set_xticks(np.arange(0, len(steps), 3)+1)
                # axs[plt_num].tick_params(top=False, right=False)

    axs[plt_num].legend()
    name_fig = TASK_PATH[0].split('/')[-1]
    fig_path = f'{dir_figs}/{name_fig}'
    if SAVE:
        plt.savefig(f'{fig_path}.pdf', bbox_inches='tight')
    else:
        pass
        # plt.show()

# ####

# Plot avg reward
SAVE = True
with plt.style.context(["grid"]):
    fig, axs = plt.subplots(1, 1, figsize=(4, 3))  # 1 plot for each env_task
    axs = np.atleast_1d(axs)  # Converts a single Axes object into a 1D array

    plt_num = 0
    max_ylim = 0
    groups = grouped_data['episode_reward']
    for group_key in groups.keys():
        print(group_key)
        if group_key in list(map(lambda x: x[0], final_hyperparams[env])):
            color = [l[1]
                     for l in final_hyperparams[env] if l[0] == group_key][0]
            # Compute mean and std of the rewards
            rews_seeds = groups[group_key]
            print(
                f'Number of files for group: {group_key}: {len(rews_seeds)}')
            # In case some exps are longer than others
            min_len = min([len(rew) for rew in rews_seeds])
            rews_seeds = [rew[:min_len] for rew in rews_seeds]

            rewards = np.array(rews_seeds)
            mean = np.mean(rewards, axis=0)
            # mean = smooth_fct(mean, kernel_size=2)
            std = np.std(rewards, axis=0)
            steps = np.arange(len(mean))+1  # add 1 because we start at 1!
            # Show only part of the curve
            steps = steps[0:10]
            mean = mean[0:len(steps)]
            std = std[0:len(steps)]

            # print(f"Group {group_key}: {mean[-1]:.2f} ± {std[-1]:.2f}")
            # Plot the mean and std
            label_ = f"ours: {group_key}" if group_key[0] == True else f"baseline: {group_key}"
            axs[plt_num].plot(
                steps, mean, label=f"{label_}", color=color, linewidth=2.0)
            axs[plt_num].fill_between(
                steps, mean - std, mean + std, color=color, alpha=0.15)
            title = env + '_avg_reward'
            axs[plt_num].set_title(title, fontsize=20)
            axs[plt_num].set_xlabel(
                f'Datasize$\\times$ {num_eval_frames}', fontsize=15)
            axs[plt_num].set_ylabel('Task reward', fontsize=15)
            max_ylim = max(max(mean), max_ylim) + 50
            if env == 'hopper':
                axs[plt_num].set_ylim([0, max_ylim])
            else:
                axs[plt_num].set_ylim([0, 1000])
            # axs[plt_num].set_xlim([0, 10])
            axs[plt_num].tick_params(axis='x', labelsize=12)
            axs[plt_num].tick_params(axis='y', labelsize=12)
            # Bold and bigger x-axis ticks
            axs[plt_num].set_xticks(np.arange(0, len(steps), 3)+1)
            # axs[plt_num].tick_params(top=False, right=False)

    axs[plt_num].legend()
    name_fig = TASK_PATH[0].split('/')[-1]
    fig_path = f'{dir_figs}/{name_fig}_avg'
    if SAVE:
        plt.savefig(f'{fig_path}.pdf', bbox_inches='tight')
    else:
        plt.show()
