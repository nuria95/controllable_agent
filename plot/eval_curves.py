import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
import scienceplots
from collections import defaultdict
import os
import yaml
import pandas as pd
plt.rcParams.update(bundles.iclr2024(
    family="serif", rel_width=0.9, nrows=1.0, ncols=1.0))
dir_figs = 'test_folder'
SAVE = True
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
}

BASE_PATH = '/home/nuria/phd/controllable_agent/results_clus'
dir_figs = '/home/nuria/phd/controllable_agent/figs'
SAVE = True
paths = [f'{BASE_PATH}/online_fb_quadruped_alltasks_vel',
         f'{BASE_PATH}/online_fb_quadruped_alltasks',
         f'{BASE_PATH}/online_fb_cheetah_alltasks',
         f'{BASE_PATH}/online_fb_maze_alltasks',
         f'{BASE_PATH}/online_fb_walker_alltasks',
         ]

grouped_files = defaultdict(list)
TASK_PATH = paths[3]
env = [e for e in list(domain_tasks.keys()) if e in TASK_PATH][0]
for exp_id in sorted(os.listdir(TASK_PATH)):

    seed_folder = os.path.join(TASK_PATH, str(exp_id))
    config_path = os.path.join(seed_folder, "config.yaml")
    eval_path = os.path.join(seed_folder, "eval.csv")

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
    # Use (uncertainty, mix_ratio) as the group key
    group_key = (uncertainty, mix_ratio) if env != 'maze' else (uncertainty, mix_ratio, add_trunk)
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


final_hyperparams = [(True, 0.3), (False, 0.3)] if env != 'maze' else [
    (True, 0.3, True), (False, 0.3, True)]
COLOR_DICT = {(True, 0.3): ourorange,
              (True, 0.3, True): ourorange,
              (False, 0.3): ourblue,
              (False, 0.3, True): ourblue
              }
with plt.style.context(["grid"]):
    fig, axs = plt.subplots(1, len(grouped_data.keys()),
                            figsize=(10, 3))  # 1 plot for each env_task
    plt_num = -1
    for env_task, groups in grouped_data.items():
        plt_num += 1
        for group_key in groups.keys():
            if group_key in final_hyperparams:
                # Compute mean and std of the rewards
                rews_seeds = groups[group_key]
                # In case some exps are longer than others
                min_len = min([len(rew) for rew in rews_seeds])
                rews_seeds = [rew[:min_len] for rew in rews_seeds]
                
                rewards = np.array(rews_seeds)
                mean = np.mean(rewards, axis=0)
                std = np.std(rewards, axis=0)
                steps = np.arange(len(mean))+1
                # print(f"Group {group_key}: {mean[-1]:.2f} ± {std[-1]:.2f}")
                # Plot the mean and std
                axs[plt_num].plot(
                    steps, mean, label=f"{group_key}", color=COLOR_DICT[group_key], linewidth=2.0)
                axs[plt_num].fill_between(
                    steps, mean - std, mean + std, color=COLOR_DICT[group_key], alpha=0.15)
                title = (' ').join(env_task.split('_')[-2::])
                axs[plt_num].set_title(title, fontsize=20)
                axs[plt_num].set_xlabel(
                    f'Datasize$\\times$ {num_eval_frames}', fontsize=15)
                axs[plt_num].set_ylabel('Task reward', fontsize=15)
                axs[plt_num].set_ylim([0, 1000])
                axs[plt_num].tick_params(axis='x', labelsize=12)
                axs[plt_num].tick_params(axis='y', labelsize=12)
                # Bold and bigger x-axis ticks
                axs[plt_num].set_xticks(np.arange(0, len(steps), 2)+1)
                # axs[plt_num].tick_params(top=False, right=False)

    axs[plt_num].legend()
    name_fig = TASK_PATH.split('/')[-1]
    fig_path = f'{dir_figs}/{name_fig}'
    if SAVE:
        plt.savefig(f'{fig_path}.pdf', bbox_inches='tight')


# with plt.style.context(["grid"]):
#   fig, ax = plt.subplots(figsize=(6, 4))
#   plt.xticks(fontsize=16, fontweight='bold')  # Bold and bigger x-axis ticks
#   plt.yticks(fontsize=16, fontweight='bold')
#   plt.tick_params(top=False, right=False)


#   ax.plot(steps, mean_b, label='heuristic', color=COLOR_DICT['heuristic'], linewidth=2.0)
#   ax.fill_between(steps, mean_b - std_b, mean_b + std_b, color=COLOR_DICT['heuristic'], alpha=0.2)

#   ax.plot(steps, mean_abl, label='CAI + learn only', color=COLOR_DICT['cailearn'], linewidth=2.0)
#   ax.fill_between(steps, mean_abl - std_abl, mean_abl + std_abl, color=COLOR_DICT['cailearn'], alpha=0.15)

#   ax.plot(steps, mean_abp, label='CAI + prior only', color=COLOR_DICT['caiprior'], linewidth=2.0)
#   ax.fill_between(steps, mean_abp - std_abp, mean_abp + std_abp, color=COLOR_DICT['caiprior'], alpha=0.15)

#   ax.plot(steps, mean_c, label='CAI + learn + prior (ours)', color=COLOR_DICT['caiman'], linewidth=2.0)
#   ax.fill_between(steps, mean_c - std_c, mean_c + std_c, color=COLOR_DICT['caiman'], alpha=0.15)
#   title = 'Single object'
#   ax.set_title(title, fontsize=20)
#   ax.set_xlabel('Iterations$\\times$100', fontsize=15)
#   ax.set_ylabel('Success Rate', fontsize=15)
#   ax.autoscale(tight=True)
#   #ax.legend(loc=4)
#   ax.grid(alpha=0.2)
#   ax.set_ylim(top=1.0)

#   name_fig = title.replace(' ', '_')
#   fig_path = f'{dir_figs}/sparse_{name_fig}'
#   if SAVE: plt.savefig(f'{fig_path}.pdf', bbox_inches = 'tight')
