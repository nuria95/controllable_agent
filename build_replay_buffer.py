from pathlib import Path
import torch
from controllable_agent import runner
from url_benchmark import goals as _goals
hp = runner.HydraEntryPoint("url_benchmark/pretrain.py")


# Potentially not needed, load it directly in train_offline! Might get deprecated!
env_name = "quadruped"
dataset_name = "rnd"
num_episodes = 1000
task_ = "quadruped_walk"
goal_space = 'simplified_quadruped'

buffer_dir = Path(f"datasets/{env_name}/{dataset_name}/buffer/")

ws = hp.workspace(task=task_, replay_buffer_episodes=num_episodes)
# Ensure goal space used for adding goals to obs, is the same as goal_space used for goals key
assert ws.cfg.goal_space == goal_space
goal_func = None if ws.cfg.goal_space is None else _goals.goal_spaces.funcs[
    ws.domain][ws.cfg.goal_space]

ws.replay_loader.load(ws.train_env, buffer_dir,
                      relabel=True, goal_func=goal_space)
with Path(f"datasets/{env_name}/{dataset_name}/replay_{num_episodes}_{goal_space}.pt").open('wb') as f:
    torch.save(ws.replay_loader, f)


# from pathlib import Path
# import torch
# from controllable_agent import runner
# hp = runner.HydraEntryPoint("url_benchmark/pretrain.py")
# # buffer_dir = Path("datasets/walker/proto/buffer/")
# # task_ = "walker_walk"

# buffer_dir = Path("datasets/point_mass_maze/proto/buffer/")
# task_ = "point_mass_maze_reach_top_left"

# ws = hp.workspace(task=task_, replay_buffer_episodes=5000)

# ws.replay_loader.load(ws.train_env, buffer_dir, relabel=True)
# new_folder = ('/').join(str(buffer_dir).split('/')[:-1]) + '/replay.pt'

# # with Path("datasets/walker/proto/replay.pt").open('wb') as f:

# with Path(new_folder).open('wb') as f:
#     torch.save(ws.replay_loader, f)
