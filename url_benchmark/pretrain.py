# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pdb  # pylint: disable=unused-import
import logging
import dataclasses
import typing as tp
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore', category=DeprecationWarning)


os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# if the default egl does not work, you may want to try:
# export MUJOCO_GL=glfw
os.environ['MUJOCO_GL'] = os.environ.get('MUJOCO_GL', 'egl')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
import torch
import wandb
import omegaconf as omgcf
# from dm_env import specs

from url_benchmark import dmc
from dm_env import specs
from url_benchmark import utils
from url_benchmark import goals as _goals
from url_benchmark.logger import Logger
from url_benchmark.in_memory_replay_buffer import ReplayBuffer
from url_benchmark.video import TrainVideoRecorder, VideoRecorder
from url_benchmark import agent as agents
from url_benchmark.d4rl_benchmark import D4RLReplayBufferBuilder, D4RLWrapper

logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True
# os.environ['WANDB_MODE']='offline'

# # # Config # # #

@dataclasses.dataclass
class Config:
    agent: tp.Any
    # misc
    seed: int = 1
    device: str = "cuda"
    save_video: bool = False
    use_tb: bool = False
    use_wandb: bool = False
    # experiment
    experiment: str = "online"
    # task settings
    task: str = "walker_stand"
    obs_type: str = "states"  # [states, pixels]
    frame_stack: int = 3  # only works if obs_type=pixels
    action_repeat: int = 1  # set to 2 for pixels
    discount: float = 0.99
    future: float = 0.99  # discount of future sampling, future=1 means no future sampling
    goal_space: tp.Optional[str] = None
    append_goal_to_observation: bool = False
    # eval
    num_eval_episodes: int = 10
    custom_reward: tp.Optional[str] = None  # activates custom eval if not None
    final_tests: int = 10
    # checkpoint
    snapshot_at: tp.Tuple[int, ...] = (100000, 200000, 500000, 800000, 1000000, 1500000,
                                       2000000, 3000000, 4000000, 5000000, 9000000, 10000000)
    checkpoint_every: int = 100000
    load_model: tp.Optional[str] = None
    # training
    num_seed_frames: int = 4000
    replay_buffer_episodes: int = 5000
    update_encoder: bool = True
    batch_size: int = omgcf.II("agent.batch_size")
    uncertainty: bool = False
    update_every_steps: int = 1
    num_agent_updates: int = 1
    warmup: bool = True
    pretrain_update_steps: int = 1000
    # to avoid hydra issues
    project_dir: str = ""
    results_dir: str = ""
    id: int = 0
    working_dir: str = ""
    debug: bool = False
    eval: bool = False
    # mode
    reward_free: bool = True
    # train settings
    num_train_frames: int = 2000010
    # snapshot
    eval_every_frames: int = 10000
    load_replay_buffer: tp.Optional[str] = None
    save_train_video: bool = False



# Name the Config as "workspace_config".
# When we load workspace_config it in the main config, we are telling it to load: Config.
ConfigStore.instance().store(name="workspace_config", node=Config)


# # # Implem # # #


def make_agent(
    obs_type: str, obs_spec, action_spec, num_expl_steps: int, cfg: omgcf.DictConfig
) -> tp.Union[agents.FBDDPGAgent]:
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = (action_spec.num_values, ) if isinstance(action_spec, specs.DiscreteArray) \
        else action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


C = tp.TypeVar("C", bound=Config)


def _init_eval_meta(workspace: "BaseWorkspace", custom_reward: tp.Optional[_goals.BaseReward] = None) -> agents.MetaDict:
    if custom_reward is not None:
        try:  # if the custom reward implements a goal, return it
            goal = custom_reward.get_goal(workspace.cfg.goal_space)
            return workspace.agent.get_goal_meta(goal)
        except Exception:  # pylint: disable=broad-exceptf
            pass
        # TODO Assuming fix episode length:
        num_steps = workspace.agent.cfg.num_inference_steps  # type: ignore
        if len(workspace.replay_loader) * workspace.replay_loader._episodes_length[0] < num_steps: 
            # print("Not enough data for inference, skipping eval")
            return None
        obs_list, reward_list = [], []
        batch_size = 0
        while batch_size < num_steps:
            batch = workspace.replay_loader.sample(workspace.cfg.batch_size, custom_reward=custom_reward)
            batch = batch.to(workspace.cfg.device)
            obs_list.append(batch.next_goal if workspace.cfg.goal_space is not None else batch.next_obs)
            reward_list.append(batch.reward)
            batch_size += batch.next_obs.size(0)
        obs, reward = torch.cat(obs_list, 0), torch.cat(reward_list, 0)  # type: ignore
        obs_t, reward_t = obs[:num_steps], reward[:num_steps]
        return workspace.agent.infer_meta_from_obs_and_rewards(obs_t, reward_t)

    if workspace.cfg.goal_space is not None:
        funcs = _goals.goals.funcs.get(workspace.cfg.goal_space, {})
        if workspace.cfg.task in funcs:
            g = funcs[workspace.cfg.task]()
            return workspace.agent.get_goal_meta(g)
    return workspace.agent.infer_meta(workspace.replay_loader)


class BaseWorkspace(tp.Generic[C]):
    def __init__(self, cfg: C) -> None:
        self.work_dir = Path.cwd() if len(cfg.working_dir) == 0 else Path(cfg.working_dir)
        self.model_dir = self.work_dir if 'cluster' not in str(self.work_dir) else Path(str(self.work_dir).replace('home', 'project/hilliges'))
        print(f'Workspace: {self.work_dir}')
        print(f'Running code in : {Path(__file__).parent.resolve().absolute()}')
        logger.info(f'Workspace: {self.work_dir}')
        logger.info(f'Running code in : {Path(__file__).parent.resolve().absolute()}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        if not torch.cuda.is_available():
            if cfg.device != "cpu":
                logger.warning(f"Falling back to cpu as {cfg.device} is not available")
                cfg.device = "cpu"
                cfg.agent.device = "cpu"
        self.device = torch.device(cfg.device)
        task = cfg.task
        if task.startswith('point_mass_maze'):
            self.domain = 'point_mass_maze'
        else:
            self.domain = task.split('_', maxsplit=1)[0]

        self.train_env = self._make_env()
        self.eval_env = self._make_env()
        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, self.domain, str(cfg.id)
            ])
            wandb.init(project="controllable_agent", group=cfg.experiment, name=exp_name,  # mode="disabled",
                       config=omgcf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), dir=self.work_dir)  # type: ignore
        if cfg.goal_space is not None:
            if cfg.goal_space not in _goals.goal_spaces.funcs[self.domain]:
                raise ValueError(f"Unregistered goal space {cfg.goal_space} for domain {self.domain}")

        self.replay_loader = ReplayBuffer(max_episodes=cfg.replay_buffer_episodes, discount=cfg.discount, future=cfg.future)

        cam_id = 0 if 'quadruped' not in self.domain else 2

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None,
                                            camera_id=cam_id, use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self.global_step = 0
        self.global_episode = 0
        self.eval_rewards_history: tp.List[float] = []
        self.eval_dist_history: tp.List[float] = []
        self._checkpoint_filepath = self.model_dir / "models" / "latest.pt"
        # This is for continuing training in case workdir is the same
        if self._checkpoint_filepath.exists():
            self.load_checkpoint(self._checkpoint_filepath)
        # This is for loading an existing model
        elif cfg.load_model is not None:
            self.load_checkpoint(cfg.load_model)  #, exclude=["replay_loader"])

        self.reward_cls: tp.Optional[_goals.BaseReward] = None
        if self.cfg.custom_reward == "maze_multi_goal":
            self.reward_cls = self._make_custom_reward(seed=self.cfg.seed)
            # Compute fix states and zs for evaluating disagreement through time
            # self.agent.eval_states = _goals.MazeMultiGoal().get_eval_states(num_states=500).to(self.device)
            self.agent.eval_states = _goals.MazeMultiGoal().get_eval_midroom_states().to(self.device)
            self.agent.eval_zs = self.agent.sample_z(len(self.agent.eval_states), device=self.device)
        else:
            self.agent.eval_states = None 

    def _make_env(self) -> dmc.EnvWrapper:
        cfg = self.cfg
        if self.domain == "d4rl":
            import d4rl  # type: ignore # pylint: disable=unused-import
            import gym
            return dmc.EnvWrapper(D4RLWrapper(gym.make(self.cfg.task.split('_')[1])))
        return dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed,
                        goal_space=cfg.goal_space, append_goal_to_observation=cfg.append_goal_to_observation)

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    def _make_custom_reward(self, seed: int) -> tp.Optional[_goals.BaseReward]:
        """Creates a custom reward function if provided in configuration
        else returns None
        """
        if self.cfg.custom_reward is None:
            return None
        return _goals.get_reward_function(self.cfg.custom_reward, seed)

    def eval_maze_goals(self) -> None:
        reward_cls = _goals.MazeMultiGoal()
        rewards = list()
        dists = list()
        successes = list()
        for g in reward_cls.goals:
            goal_rewards = list()
            goal_distances = list()
            goal_successes = list()
            meta = self.agent.get_goal_meta(g)
            for episode in range(self.cfg.num_eval_episodes):
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                time_step = self.eval_env.reset()
                episode_reward = 0.0
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                meta,
                                                0,
                                                eval_mode=True)
                    time_step = self.eval_env.step(action)
                    self.video_recorder.record(self.eval_env)
                    assert isinstance(time_step, dmc.ExtendedGoalTimeStep)
                    step_reward, distance, success = reward_cls.from_goal(time_step.goal, g)
                    episode_reward += step_reward
                goal_rewards.append(episode_reward)
                goal_distances.append(float(distance))
                goal_successes.append(success)
                self.video_recorder.save(f'{g}.mp4')
            print(f"goal: {g}, avg_reward: {round(float(np.mean(goal_rewards)), 2)}, "
                  f"avg_distance: {round(float(np.mean(goal_distances)), 5)}, "
                  f"avg_success: {round(float(np.mean(goal_successes)), 5)}")
            rewards.append(float(np.mean(goal_rewards)))
            dists.append(float(np.mean(goal_distances)))
            successes.append(float(np.mean(goal_successes)))  # num goals x 1
        self.eval_rewards_history.append(float(np.mean(rewards)))
        self.eval_dist_history.append(float(np.mean(dists)))
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', self.eval_rewards_history[-1])
            log('episode_distance', self.eval_dist_history[-1])
            log('step', self.global_step)
            log('episode', self.global_episode)
            log('success_rate', float(np.mean(successes)))
            for i, room in zip(range(0, len(successes), reward_cls.goals_per_room), range(1, 5)):
                log(f'success_room{room}', float(np.mean(successes[i:i+reward_cls.goals_per_room])))

    def eval(self) -> None:
        step, episode = 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        physics_agg = dmc.PhysicsAggregator()
        rewards: tp.List[float] = []
        normalized_scores: tp.List[float] = []
        # For goal-reaching tasks (goal space + no custom_reward)
        if self.cfg.goal_space is not None and self.cfg.custom_reward is None:
            meta = _init_eval_meta(self)
        z_correl = 0.0
        is_d4rl_task = self.cfg.task.split('_')[0] == 'd4rl'
        actor_success: tp.List[float] = []
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            # create custom reward if need be (if field exists)
            seed = 12 * self.cfg.num_eval_episodes + len(rewards)
            custom_reward = self._make_custom_reward(seed=seed)
            if custom_reward is not None:
                meta = _init_eval_meta(self, custom_reward)
                if meta is None:  # not enough data to perform z inference
                    return
            total_reward = 0.0
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                physics_agg.add(self.eval_env)
                self.video_recorder.record(self.eval_env)
                if self.agent.cfg.additional_metric:
                    z_correl += self.agent.compute_z_correl(time_step, meta)
                    actor_success.extend(self.agent.actor_success)
                if custom_reward is not None:
                    time_step.reward = custom_reward.from_env(self.eval_env)
                total_reward += time_step.reward
                step += 1
            if is_d4rl_task:
                normalized_scores.append(self.eval_env.get_normalized_score(total_reward))
            rewards.append(total_reward)
            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        self.eval_rewards_history.append(float(np.mean(rewards)))
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            if is_d4rl_task:
                log('episode_normalized_score', float(100 * np.mean(normalized_scores)))
            log('episode_reward', self.eval_rewards_history[-1])
            if len(rewards) > 1:
                log('episode_reward#std', float(np.std(rewards)))
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('z_correl', z_correl / episode)
            log('step', self.global_step)
            if actor_success:
                log('actor_sucess', float(np.mean(actor_success)))
            log('z_norm', np.linalg.norm(meta['z']).item())
            for key, val in physics_agg.dump():
                log(key, val)

    _CHECKPOINTED_KEYS = ('agent', 'global_step', 'global_episode', "replay_loader")

    def save_checkpoint(self, fp: tp.Union[Path, str], exclude: tp.Sequence[str] = ()) -> None:
        logger.info(f"Saving checkpoint to {fp}")
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True, parents=True)
        assert isinstance(self.replay_loader, ReplayBuffer), "Is this buffer designed for checkpointing?"
        # this is just a dumb security check to not forget about it
        payload = {k: self.__dict__[k] for k in self._CHECKPOINTED_KEYS if k not in exclude}
        with fp.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_checkpoint(self, fp: tp.Union[Path, str], only: tp.Optional[tp.Sequence[str]] = None, exclude: tp.Sequence[str] = ()) -> None:
        """Reloads a checkpoint or part of it

        Parameters
        ----------
        only: None or sequence of str
            reloads only a specific subset (defaults to all)
        exclude: sequence of str
            does not reload the provided keys
        """
        print(f"loading checkpoint from {fp}")
        fp = Path(fp)
        with fp.open('rb') as f:
            payload = torch.load(f)
        if isinstance(payload, ReplayBuffer):  # compatibility with pure buffers pickles
            payload = {"replay_loader": payload}
        if only is not None:
            only = list(only)
            assert all(x in self._CHECKPOINTED_KEYS for x in only)
            payload = {x: payload[x] for x in only}
        exclude = list(exclude)
        assert all(x in self._CHECKPOINTED_KEYS for x in exclude)
        for x in exclude:
            payload.pop(x, None)
        for name, val in payload.items():
            logger.info("Reloading %s from %s", name, fp)
            if name == "agent":
                self.agent.init_from(val)
            elif name == "replay_loader":
                assert isinstance(val, ReplayBuffer)
                # pylint: disable=protected-access
                # drop unecessary meta which could make a mess
                val._current_episode.clear()  # make sure we can start over
                val._future = self.cfg.future
                val._discount = self.cfg.discount
                # val._max_episodes = len(val._storage["discount"])
                # val._idx = len(val._storage["discount"]) % self.cfg.replay_buffer_episodes
                # val._full = val._idx == 0
                val._max_episodes = self.cfg.replay_buffer_episodes
                val._episodes_length = np.array([len(array) - 1 for array in val._storage["discount"]], dtype=np.int32)
                self.replay_loader = val
                # # TODO:  This leads to out of RAM for now (to be fixed)
                # if not self.replay_loader._full:  # if buffer is not full we need to recreate the storage
                #     self.replay_loader.prefill(val._storage)
            else:
                assert hasattr(self, name)
                setattr(self, name, val)
                if name == "global_episode":
                    logger.warning(f"Reloaded agent at global episode {self.global_episode}")

    def finalize(self) -> None:
        print("Running final test", flush=True)
        repeat = self.cfg.final_tests
        if not repeat:
            return

        if self.cfg.custom_reward == "maze_multi_goal":
            eval_hist = self.eval_rewards_history
            rewards = {}
            self.eval_rewards_history = []
            self.cfg.num_eval_episodes = repeat
            self.eval_maze_goals()
            rewards["rewards"] = self.eval_rewards_history
            self.eval_rewards_history = eval_hist  # restore
        else:
            domain_tasks = {
                "cheetah": ['walk', 'walk_backward', 'run', 'run_backward'],
                "quadruped": ['stand', 'walk', 'run', 'jump'],
                "walker": ['stand', 'walk', 'run', 'flip', 'upside'],
            }
            if self.domain not in domain_tasks:
                return
            eval_hist = self.eval_rewards_history
            rewards = {}
            for name in domain_tasks[self.domain]:
                task = "_".join([self.domain, name])
                self.cfg.task = task
                self.cfg.custom_reward = task  # for the replay buffer
                self.cfg.seed += 1  # for the sake of avoiding similar seeds
                self.eval_env = self._make_env()
                self.eval_rewards_history = []
                self.cfg.num_eval_episodes = 1
                for _ in range(repeat):
                    self.eval()
                rewards[task] = self.eval_rewards_history
        self.eval_rewards_history = eval_hist  # restore
        with (self.work_dir / "test_rewards.json").open("w") as f:
            json.dump(rewards, f)


class Workspace(BaseWorkspace[Config]):
    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if cfg.save_train_video else None,
                                                       camera_id=self.video_recorder.camera_id, use_wandb=self.cfg.use_wandb)
        if not self._checkpoint_filepath.exists():  # don't relay if there is a checkpoint
            if cfg.load_replay_buffer is not None:
                if self.cfg.task.split('_')[0] == "d4rl":
                    d4rl_replay_buffer_builder = D4RLReplayBufferBuilder()
                    self.replay_storage = d4rl_replay_buffer_builder.prepare_replay_buffer_d4rl(self.train_env, self.agent.init_meta(), self.cfg)
                    self.replay_loader = self.replay_storage
                else:
                    self.load_checkpoint(cfg.load_replay_buffer, only=["replay_loader"])
                if not cfg.warmup:
                    print('\nNot warming up when loading a replay buffer!!!\n')
            else:
                assert not cfg.warmup, "Trying to warmup without a preloaded replay buffer"

    def _init_meta(self, obs: np.ndarray = None):
        meta = self.agent.init_meta(obs)
        return meta

    def train(self) -> None:
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        update_every_step = utils.Every(self.agent.cfg.update_every_steps,
                                        self.cfg.action_repeat)
        episode_step, episode_reward, z_correl = 0, 0.0, 0.0
        time_step = self.train_env.reset()
        meta = self._init_meta(time_step.observation)
        self.replay_loader.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        meta_disagr = []
        physics_agg = dmc.PhysicsAggregator()

        while train_until_step(self.global_step):
            # try to update the agent
            if not seed_until_step(self.global_step) and update_every_step(self.global_step):
                if self.global_step == 0 and self.cfg.warmup:
                    print("Pretraining...")
                    for _ in range(self.cfg.pretrain_update_steps):
                        metrics = self.agent.update(self.replay_loader, self.global_step)
                    print('\nPretraining done\n')
                for _ in range(self.cfg.num_agent_updates):
                    metrics = self.agent.update(self.replay_loader, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if time_step.last():
                self.global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_loader))
                        log('step', self.global_step)
                        log('z_correl', z_correl)
                        if self.cfg.uncertainty and len(meta_disagr) > 0:
                            log('z_disagr', np.mean(meta_disagr))

                        for key, val in physics_agg.dump():
                            log(key, val)
                # reset env
                time_step = self.train_env.reset()
                meta = self._init_meta(time_step.observation)
                self.replay_loader.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                # if self.global_frame in self.cfg.snapshot_at:
                #     self.save_checkpoint(self._checkpoint_filepath.with_name(f'snapshot_{self.global_frame}.pt'))
                episode_step = 0
                episode_reward = 0.0
                z_correl = 0.0
                meta_disagr = []

            # try to evaluate
            if eval_every_step(self.global_step) and not self.cfg.debug:
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.cfg.custom_reward == "maze_multi_goal":
                    self.eval_maze_goals()
                else:
                    self.eval()
            meta = self.agent.update_meta(meta, self.global_step, time_step, finetune=False, replay_loader=self.replay_loader,
                                          obs=time_step.observation)
            if self.cfg.uncertainty and 'disagr' in meta and meta['updated']:
                meta_disagr.append(meta['disagr'])
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # take env step
            time_step = self.train_env.step(action)
            physics_agg.add(self.train_env)
            episode_reward += time_step.reward
            self.replay_loader.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            z_correl += self.agent.compute_z_correl(time_step, meta)
            episode_step += 1
            self.global_step += 1
            # save checkpoint to reload
            if not self.global_frame % self.cfg.checkpoint_every:
                self.save_checkpoint(self._checkpoint_filepath)
        self.save_checkpoint(self._checkpoint_filepath)  # make sure we save the final checkpoint
        self.finalize()

    def eval_model(self) -> None:
        self.eval()
        if 'maze' not in self.cfg.task:
            return
        self.agent.compute_disagreement_metrics()
        xy = self.agent.eval_states[:, :2].cpu().numpy()  # num_states x 2
        # self.agent.Q1 is num_ensembles x num_states
        ep_std1, ep_std2 = self.agent.Q1.std(dim=0).cpu().numpy(), self.agent.Q2.std(dim=0).cpu().numpy()  # num_states
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(xy[:, 0], xy[:, 1], c=ep_std1, cmap='viridis', s=100, edgecolor='k')
        plt.colorbar(scatter, label="Value")  # Add a colorbar
        plt.title("Q1 std")
        plt.show()


@hydra.main(config_path='configs', config_name='base_config', version_base="1.1")
def main(cfg: omgcf.DictConfig) -> None:
    # we assume cfg is a PretrainConfig (but actually not really)
    # calls Config and PretrainConfig
    workspace = Workspace(cfg)  # type: ignore
    if not cfg.eval:
        workspace.train()
    else:
        workspace.eval_model()


if __name__ == '__main__':
    main()
