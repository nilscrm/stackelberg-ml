

import csv
import pathlib
from typing import List
import warnings
import gymnasium
import numpy as np
import torch
from tqdm import tqdm
from stackelberg_mbrl.envs.querying_env import ConstantContextEnv, CountedEnvWrapper
from stackelberg_mbrl.experiments.experiment_config import TableWorldModelConfig, SampleEfficiency, ExperimentConfig
from stackelberg_mbrl.policies.random_policy import RandomPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from stackelberg_mbrl.util.trajectories import sample_trajectories


def take(samples: List[torch.Tensor], max_samples: int, vec_size: int):
    out = []
    count = 0
    for sample in samples:
        mask = sample[vec_size:]
        add_count = torch.sum(mask)
        if add_count + count > max_samples: break
        out.append(sample)
        count += add_count
    return out


def Tabular(env_true: gymnasium.Env, oracle: ActorCriticPolicy, config: ExperimentConfig):
    model_config = config.world_model_config
    
    ns = env_true.num_states
    na = env_true.num_actions
    vec_size = ns * na * ns

    rnd_policy = RandomPolicy(ns, na, 0)
    samples: List[torch.Tensor] = []
    model = torch.full((ns*na*ns,), 1/ns)
    old_model = torch.full_like(model, -np.inf)

    if config.sample_efficiency is not None:
        sample_counter = TabularSampleCounter(oracle=oracle, eval_env=env_true, config=config.sample_efficiency)

    if config.sample_efficiency.max_samples > 0:
        warnings.warn("max_samples is not used in the tabular world model")

    env_true_count = CountedEnvWrapper(env_true)
    for i in tqdm(range(model_config.max_training_samples + model_config.init_sample_trajectories)):
        if i == 0 and model_config.init_sample_trajectories > 0:
            traj = sample_trajectories(env_true_count, rnd_policy, model_config.init_sample_trajectories, context=model, w_one_hot=False)
        elif np.random.rand() < model_config.noise:
            traj = sample_trajectories(env_true_count, rnd_policy, model_config.batch_size, context=model, w_one_hot=False)
        else:
            traj = sample_trajectories(env_true_count, oracle, model_config.batch_size, context=model, w_one_hot=False)
        query, mask = traj.to_query_tensor()
        samples.append(torch.concat((query.reshape(-1), mask.reshape(-1))))
        
        old_model = model
        
        sx = torch.stack(take(samples, model_config.max_training_samples, vec_size))
        model = torch.sum(sx[:, :vec_size].reshape(-1, ns*na, ns) * sx[:, vec_size:, None], dim=0) / (sx[:, vec_size:].sum(dim=0)[:, None] + 1e-20)
        model = model.reshape(-1)

        if config.sample_efficiency is not None:
            sample_counter.step(env_true_count.samples, model)

        if torch.norm(model - old_model) < model_config.eps: break
        if env_true_count.samples > model_config.max_training_samples: break

    if config.sample_efficiency is not None and config.sample_efficiency.log_save_name is not None:
        sample_counter.save(config.output_dir / config.experiment_name / "sample_efficiency" / config.sample_efficiency.log_save_name)

    return model, sample_counter.evals if config.sample_efficiency is not None else None

class TabularSampleCounter():
    def __init__(self, oracle, eval_env, config: SampleEfficiency) -> None:
        self.evals = []
        self.next_eval = 0
        self.oracle = oracle
        self.eval_env = eval_env
        self.config = config

    def step(self, samples, model):
        if samples < self.next_eval: return
        self.next_eval += self.config.sample_eval_rate
        env = ConstantContextEnv(self.eval_env, model.reshape(-1))
        r_mean, r_std = evaluate_policy(self.oracle, env, n_eval_episodes=self.config.n_eval_episodes)
        self.evals.append((samples, r_mean))

    def save(self, filename):
        filename = pathlib.Path(filename)
        if not filename.parent.exists(): filename.parent.mkdir(parents=True)
        with open(filename, 'w', newline='') as file:
            sw = csv.writer(file)
            sw.writerows(self.evals)
