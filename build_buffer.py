from pathlib import Path
import torch
from controllable_agent import runner
hp = runner.HydraEntryPoint("url_benchmark/pretrain.py")
# buffer_dir = Path("datasets/walker/proto/buffer/")
# task_ = "walker_walk"

buffer_dir = Path("datasets/point_mass_maze/proto/buffer/")
task_ = "point_mass_maze_reach_top_left"

ws = hp.workspace(task=task_, replay_buffer_episodes=5000)

ws.replay_loader.load(ws.train_env, buffer_dir, relabel=True)
new_folder = ('/').join(str(buffer_dir).split('/')[:-1]) + '/replay.pt'

# with Path("datasets/walker/proto/replay.pt").open('wb') as f:

with Path(new_folder).open('wb') as f:
    torch.save(ws.replay_loader, f)