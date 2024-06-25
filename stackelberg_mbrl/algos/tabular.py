

from typing import List
import gymnasium
import numpy as np
import torch
from tqdm import tqdm
from stackelberg_mbrl.envs.querying_env import CountedEnvWrapper
from stackelberg_mbrl.experiments.experiment_config import TableWorldModelConfig
from stackelberg_mbrl.policies.random_policy import RandomPolicy
from stable_baselines3.common.policies import ActorCriticPolicy

from stackelberg_mbrl.util.trajectories import sample_trajectories


def Tabular(env_true: gymnasium.Env, oracle: ActorCriticPolicy, config: TableWorldModelConfig):
    ns = env_true.num_states
    na = env_true.num_actions
    vec_size = ns * na * ns

    rnd_policy = RandomPolicy(ns, na, 0)
    samples: List[torch.Tensor] = []
    model = torch.full((ns*na*ns,), 1/ns)
    old_model = torch.full_like(model, -np.inf)

    env_true_count = CountedEnvWrapper(env_true)
    for i in tqdm(range(config.max_training_samples + config.init_sample_trajectories)):
        if config.init_sample_trajectories > 0:
            traj = sample_trajectories(env_true_count, rnd_policy, config.init_sample_trajectories, context=model, w_one_hot=False)
        elif np.random.rand() < config.noise:
            traj = sample_trajectories(env_true_count, rnd_policy, config.batch_size, context=model, w_one_hot=False)
        else:
            traj = sample_trajectories(env_true_count, oracle, config.batch_size, context=model, w_one_hot=False)
        query, mask = traj.to_query_tensor()
        samples.append(torch.concat((query.reshape(-1), mask.reshape(-1))))
        
        old_model = model
        
        sx = torch.stack(samples)
        model = torch.sum(sx[:, :vec_size].reshape(-1, ns*na, ns) * sx[:, vec_size:, None], dim=0) / (sx[:, vec_size:].sum(dim=0)[:, None] + 0.0001)
        model = model.reshape(-1)
        if torch.norm(model - old_model) < config.eps: break
        if env_true_count.samples > config.max_training_samples: break

    return model