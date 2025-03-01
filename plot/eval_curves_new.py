import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles
import scienceplots
from collections import defaultdict
import os
import yaml
import pandas as pd


def get_label(group_key):
    style_ = 'solid'
    if group_key == 'rnd':  # rnd offlien buffer
        label_ = 'rnd_buffer_agent'
        style_ = 'dotted'
        return label_, style_
    if group_key[0] == True:
        label_ = 'ours'
        if group_key[-3] == True:
            label_ += ' sampling'
            if group_key[-2] == 0:
                label_ += ' no_update'
                style_ = 'dashed'
        else:
            label_ += ' policy'
    else:
        label_ = 'baseline: uniform'
        if group_key[-2] == 0:
            label_ += ' no_update'
            style_ = 'dashed'

    return label_, style_


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
ourdarkblue = (0.368, 0.607, 0.9)
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
    # 'hop_backward', 'flip', 'flip_backward'],
    "hopper": ['hop', 'stand', 'flip'],  # , 'hop_backward', 'flip_backward'],
    "ball_in_cup": ['catch']
}

BASE_PATH = '/home/nuria/phd/controllable_agent/results_clus'

# group_key = (uncertainty, mix_ratio, add_trunk, update_z_every, sampling, update_z_proba, expl_strategy)
final_hyperparams = {'hopper': [

    [(True, 0.3, None, 100, True, 0., None), ourorange],
    # [(True, 0.3, None, 100, False, 1.), ourgreen],
    [(False, 0.3, None, 100, False, 1., None), ourblue],
    [(False, 0.3, None, 100, False, 0., None), ourblue],
    [(True, 0.3, None, 100, True, 1., None), ourorange],
],

    'maze': [
    [(True, 0.3, True, 100, True, 0., None), ourorange],
    #   [(True, 0.3, True, 100, False, 1.), ourgreen],
    [(False, 0.3, True, 100, False, 1., None), ourblue],
    [(False, 0.3, True, 100, False, 0., None), ourblue],
    [(True, 0.3, True, 100, True, 1., None), ourorange],


],


    'cheetah': [
    [(True, 0.3, None, 100, True, 0., None), ourorange],
    #  [(True, 0.3, None, 100, False, 1.), ourgreen],
    [(False, 0.3, None, 100, False, 1., None), ourblue],
    [(False, 0.3, None, 100, False, 0., None), ourblue],
    [(True, 0.3, None, 100, True, 1., None), ourorange],


],

    'quadruped': [
    [(True, 0.3, None, 100, True, 0. ,None), ourorange],
    #    [(True, 0.3, None, 100, False, 1.), ourgreen],
    [(False, 0.3, None, 100, False, 1., None), ourblue],
    [(False, 0.3, None, 100, False, 0., None), ourblue],
    [(True, 0.3, None, 100, True, 1., None), ourorange],


],

    'walker': [
    [(True, 0.3, None, 100, True, 0., None), ourorange],
    # [(True, 0.3, None, 100, False, 1.), ourgreen],
    [(False, 0.3, None, 100, False, 1., None), ourblue],
    [(False, 0.3, None, 100, False, 0., None), ourblue],
    [(True, 0.3, None, 100, True, 1., None), ourorange],


],

#     'ball_in_cup': [
#     [(True, 0.3, None, 100, True, 1.), ourorange],
#     [(True, 0.3, None, 100, True, 0.), ourdarkred],
#     #  [(True, 0.3, None, 100, False, 1.), ourgreen],
#     [(False, 0.3, None, 100, False, 1.), ourblue],
#     [(False, 0.3, None, 100, False, 0.), ourblue]
# ]

}

# group_key = (expl_strategy, mix_ratio)
final_hyperparams_random = [('act_rand', 0.3), ourdarkgreen]

dir_figs = '/home/nuria/phd/controllable_agent/figs/exp6'
paths = [f'{BASE_PATH}/quadruped',
        #  f'{BASE_PATH}/quadruped_zprobab',  # rerunning longer
         f'{BASE_PATH}/quadruped_zprobab2',
         f'{BASE_PATH}/offline_rnd_quadruped',

         f'{BASE_PATH}/maze3',
         f'{BASE_PATH}/maze3_zprobab',
         f'{BASE_PATH}/offline_rnd_maze2',

         f'{BASE_PATH}/walker',
         f'{BASE_PATH}/walker_zprobab2',
         f'{BASE_PATH}/offline_rnd_walker',


         f'{BASE_PATH}/hopper',
         f'{BASE_PATH}/hopper_zprobab',
         #  f'{BASE_PATH}/offline_rnd_hopper', #no buffer for hopper



         #  f'{BASE_PATH}/fb_ball_in_cup',
         f'{BASE_PATH}/cheetah2',
         f'{BASE_PATH}/cheetah_zprobab',
         f'{BASE_PATH}/offline_rnd_cheetah',
         
         f'{BASE_PATH}/quadruped_a_exploration',
         f'{BASE_PATH}/maze_a_exploration',
         f'{BASE_PATH}/walker_a_exploration',
         f'{BASE_PATH}/hopper_a_exploration',
         f'{BASE_PATH}/cheetah_a_exploration',
         f'{BASE_PATH}/humanoid_a_exploration',


         ]

TASK_PATHS = [
    [paths[0], paths[1], paths[2], paths[14]],  # quadruped
    [paths[3], paths[4], paths[5], paths[15]],  # maze
    [paths[6], paths[7], paths[8], paths[16]],  # walker
    [paths[9], paths[10], paths[17], ],  # hopper
    [paths[11], paths[12], paths[13], paths[18] ],  # cheetah
]
yaxis_cut = True

for TASK_PATH in TASK_PATHS:
    ###

    grouped_files = defaultdict(list)

    env = [e for e in list(domain_tasks.keys()) if e in TASK_PATH[0]][0]
    print(f'*****\n\n\nEnv {env}******\n\n')
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

        update_z_proba = config.get("agent").get(
            "update_z_proba", 1.)  # Default to None if missing

        rnd_buffer_agent = config.get("expl_agent", False)
        
        expl_strategy = config.get("agent").get("expl_strategy", None)

        # Use (uncertainty, mix_ratio) as the group key
        group_key = (uncertainty, mix_ratio, add_trunk, update_z_every,
                     sampling, update_z_proba, expl_strategy)  # if env != 'maze' else (uncertainty, mix_ratio, add_trunk)

        group_key_rnd = (rnd_buffer_agent)
        
        group_key_random = (expl_strategy, mix_ratio)
        num_eval_frames = config.get("eval_every_frames")
        # Store the eval.csv file path in the corresponding group
        # Dict with keys the different set of params, and values the list of files with same params
        grouped_files[group_key].append(eval_path)
        grouped_files[group_key_rnd].append(eval_path)
        grouped_files[group_key_random].append(eval_path)

    # Dict of dicts. Keys are task names, values are dictionary of group name and value the sequence of rewards
    grouped_data = defaultdict(dict)
    for task in domain_tasks[env]:
        key_rew = f"episode_reward_{env}_{task}" if env != 'maze' else f"reward_{task}"
        grouped_data[key_rew] = defaultdict(dict)

        for group_key, paths in grouped_files.items():
            grouped_data[key_rew][group_key] = list()
            for path in paths:
                # print(f"Group {group_key}, {key_rew}: {path}")
                df = pd.read_csv(path)
                try:
                    rewards = df[key_rew].tolist()  # Convert column to list
                except:
                    print(f'Omitting {key_rew} for {path}')
                    pass
                # Store data in the grouped dictionary
                grouped_data[key_rew][group_key].append(rewards)

    key_rew = 'episode_reward'
    # Adding avg reward among all tasks
    for group_key, paths in grouped_files.items():
        grouped_data[key_rew][group_key] = list()
        for path in paths:
            # print(f"Group {group_key}, {key_rew}: {path}")
            df = pd.read_csv(path)
            try:
                rewards = df[key_rew].tolist()  # Convert column to list
            except:
                print(f'Omitting {key_rew} for {path}')
                pass
            # Store data in the grouped dictionary
            grouped_data[key_rew][group_key].append(rewards)

    ##########################################
    # # Plot reward per task
    SAVE = True
    with plt.style.context(["grid"]):
        fig, axs = plt.subplots(1, len(grouped_data.keys())-1,
                                figsize=(10, 3))  # 1 plot for each env_task
        plt_num = -1
        for env_task, groups in grouped_data.items():
            if "episode_reward" == env_task:  # dunnot plot avg when showing all tasks
                continue
            plt_num += 1
            max_ylim = 0

            for group_key in groups.keys():

                if group_key in list(map(lambda x: x[0], final_hyperparams[env])) or group_key in final_hyperparams_random:
                    if group_key in list(map(lambda x: x[0], final_hyperparams[env])):
                        color = [l[1]
                                for l in final_hyperparams[env] if l[0] == group_key][0]
                    else:
                        color = final_hyperparams_random[1]
                    # Compute mean and std of the rewards
                    rews_seeds = groups[group_key]
                    print(
                        f'Number of files for group: {group_key}: {len(rews_seeds)}')
                    # In case some exps are longer than others\
                    # min_len = min([len(rew) for rew in rews_seeds])
                    # rews_seeds = [rew[:min_len] for rew in rews_seeds]
                    rews_seeds = [rew for rew in rews_seeds if len(rew) > 9]
                    if len(rews_seeds) < 7:
                        print(f'Removed too many ALL TASKS runs! {len(rews_seeds)}, {env}')
                        

                    rewards = np.array(rews_seeds)
                    mean = np.mean(rewards, axis=0)
                    # mean = smooth_fct(mean, kernel_size=2)
                    std = np.std(rewards, axis=0)
                    # add 1 because we start at 1!
                    if env != 'maze':
                        mean = mean[0:10]
                        std = std[0:10]
                    else:
                        # because we do an eval at timestep 0!
                        mean = mean[1:12]
                        std = std[1:12]

                    # add 1 because we start at 1!
                    steps = np.arange(len(mean))+1

                    # print(f"Group {group_key}: {mean[-1]:.2f} ± {std[-1]:.2f}")
                    # Plot the mean and std
                    # label_ = f"ours: {group_key}" if group_key[0] == True else f"baseline: {group_key}"

                    label_, linestyle_ = get_label(group_key)

                    axs[plt_num].plot(
                        steps, mean, label=f"{label_}", color=color, linewidth=2.0, linestyle=linestyle_)
                    axs[plt_num].fill_between(
                        steps, mean - std, mean + std, color=color, alpha=0.15)

                    max_ylim = max(max(mean), max_ylim)

                elif group_key == 'rnd':  # rnd_buffer_agent
                    color = 'k'
                    # Compute mean and std of the rewards
                    rews_seeds = groups[group_key]
                    print(
                        f'Number of files for group: {group_key}: {len(rews_seeds)}')
                    # In case some exps are longer than others
                    # min_len = min([len(rew) for rew in rews_seeds])
                    
                    rews_seeds = [rew for rew in rews_seeds if len(
                        rew) > 36] if env!='maze' else [rew for rew in rews_seeds if len(
                        rew) > 8] # remove short runs. For offline_maze2 we evaluated every 100k, for offline_maze every 10k
                    if len(
                        rews_seeds) < 7:
                        print(f'Removed too many RND runs! {len(rews_seeds)}, {env}')
                        continue
                    min_len = min([len(rew) for rew in rews_seeds])
                    rews_seeds = [rew[:min_len] for rew in rews_seeds]

                    rewards = np.array(rews_seeds)
                    print('Num of grads steps x 1000', len(rewards[0]))
                    mean = np.mean(rewards, axis=0)
                    max_ylim = max(max(mean), max_ylim)
                    label_, linestyle_ = get_label(group_key)
                    steps = np.arange(0,11)
                    rnd_topline = [mean[-1]] * len(steps)
                    axs[plt_num].plot(
                        steps, rnd_topline, label=f"{label_}", color=color, linewidth=2.0, linestyle=linestyle_)

            # Adjust plot per each task nicer
            title = (' ').join(env_task.replace('episode_reward_', '').split(
                '_')) if env != 'maze' else 'maze ' + env_task.replace('reward_', '')
            axs[plt_num].set_title(title, fontsize=20)
            axs[plt_num].set_xlabel(
                f'Environment steps$\\times 10^5$ ', fontsize=15)
            axs[plt_num].set_ylabel('Task reward', fontsize=15)
            if yaxis_cut:
                axs[plt_num].set_ylim([0, max_ylim + 50])
            else:
                axs[plt_num].set_ylim([0, 1000])
            axs[plt_num].set_xlim([0, 10])
            axs[plt_num].tick_params(axis='x', labelsize=12)
            axs[plt_num].tick_params(axis='y', labelsize=12)
            # Bold and bigger x-axis ticks
            axs[plt_num].set_xticks(np.arange(0, len(steps)+1, 2))
            # axs[plt_num].tick_params(top=False, right=False)
            axs[plt_num].grid(True, linestyle='-', color='grey')  # Continuous lines, grey color

        # axs[plt_num].legend()
        yaxis = 'yaxiscut' if yaxis_cut else ''
        name_fig = TASK_PATH[0].split('/')[-1] + yaxis
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
        # 1 plot for each env_task
        fig, axs = plt.subplots(1, 1, figsize=(4, 3))
        # Converts a single Axes object into a 1D array
        axs = np.atleast_1d(axs)

        plt_num = 0
        max_ylim = 0
        groups = grouped_data['episode_reward']
        for group_key in groups.keys():
            if group_key in list(map(lambda x: x[0], final_hyperparams[env])) or group_key in final_hyperparams_random:
                if group_key in list(map(lambda x: x[0], final_hyperparams[env])):
                    color = [l[1]
                            for l in final_hyperparams[env] if l[0] == group_key][0]
                else:
                    color = final_hyperparams_random[1]
                # Compute mean and std of the rewards
                rews_seeds = groups[group_key]
                print(
                    f'Number of files for group: {group_key}: {len(rews_seeds)}')
                # In case some exps are longer than others
                # min_len = min([len(rew) for rew in rews_seeds])
                rews_seeds = [rew for rew in rews_seeds if len(rew) > 9]
                assert len(
                    rews_seeds) > 7, f'Removed too many AVG runs! {len(rews_seeds)}, {env}'
                # rews_seeds = [rew[:min_len] for rew in rews_seeds]

                rewards = np.array(rews_seeds)
                mean = np.mean(rewards, axis=0)
                # mean = smooth_fct(mean, kernel_size=2)
                std = np.std(rewards, axis=0)
                # Show only part of the curve
                if env != 'maze':
                    mean = mean[0:10]
                    std = std[0:10]
                else:
                    mean = mean[1:12]  # because we do an eval at timestep 0!
                    std = std[1:12]

                steps = np.arange(len(mean))+1  # add 1 because we start at 1!

                # print(f"Group {group_key}: {mean[-1]:.2f} ± {std[-1]:.2f}")
                # Plot the mean and std
                label_, linestyle_ = get_label(group_key)
                axs[plt_num].plot(
                    steps, mean, label=f"{label_}", color=color, linewidth=2.0, linestyle=linestyle_)
                axs[plt_num].fill_between(
                    steps, mean - std, mean + std, color=color, alpha=0.15)
                title = env
                max_ylim = max(max(mean), max_ylim)

            elif group_key == 'rnd':  # rnd_buffer_agent
                color = 'k'
                # Compute mean and std of the rewards
                rews_seeds = groups[group_key]
                print(
                    f'Number of files for group: {group_key}: {len(rews_seeds)}')
                # In case some exps are longer than others
                # min_len = min([len(rew) for rew in rews_seeds])
                rews_seeds = [rew for rew in rews_seeds if len(
                    rew) > 36]  if env!='maze' else [rew for rew in rews_seeds if len(
                    rew) > 8] # remove short runs. For offline_maze2 we evaluated every 100k, for offline_maze every 10k
                if len(
                    rews_seeds) < 7:
                    print(f'Removed too many RND AVG runs! {len(rews_seeds)}, {env}')
                min_len = min([len(rew) for rew in rews_seeds])
                rews_seeds = [rew[:min_len] for rew in rews_seeds]

                rewards = np.array(rews_seeds)
                print('Num of grads steps x 1000', len(rewards[0]))
                mean = np.mean(rewards, axis=0)
                max_ylim = max(max(mean), max_ylim)
                label_, linestyle_ = get_label(group_key)
                steps = np.arange(0,11)
                rnd_topline = [mean[-1]] * len(steps)
                axs[plt_num].plot(
                    steps, rnd_topline, label=f"{label_}", color=color, linewidth=2.0, linestyle=linestyle_)

        # Adjust the plot
        if yaxis_cut:
            axs[plt_num].set_ylim([0, max_ylim + 50])
        else:
            axs[plt_num].set_ylim([0, 1000])
        axs[plt_num].set_xlim([0, 10])
        axs[plt_num].tick_params(axis='x', labelsize=12)
        axs[plt_num].tick_params(axis='y', labelsize=12)
        # Bold and bigger x-axis ticks
        axs[plt_num].set_xticks(np.arange(0, len(steps)+1,2))
        # axs[plt_num].tick_params(top=False, right=False)
        axs[plt_num].grid(True, linestyle='-', color='grey')  # Continuous lines, grey color
        axs[plt_num].set_title(title, fontsize=20)
        axs[plt_num].set_xlabel(
            f'Environment steps$\\times 10^5$ ', fontsize=15)
        axs[plt_num].set_ylabel('Task reward', fontsize=15)
        # axs[plt_num].legend()
        yaxis = 'yaxiscut' if yaxis_cut else ''
        name_fig = TASK_PATH[0].split('/')[-1] + yaxis
        fig_path = f'{dir_figs}/{name_fig}_avg'
        if SAVE:
            plt.savefig(f'{fig_path}.pdf', bbox_inches='tight')
        else:
            plt.show()
