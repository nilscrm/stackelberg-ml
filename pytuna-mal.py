import stackelberg_mbrl.envs.simple_mdp
import gymnasium
import optuna
from optuna.storages import JournalFileStorage
import numpy as np
from itertools import product
from stable_baselines3.ppo import PPO
import torch
from stackelberg_mbrl.train_mal import train_contextualized_MAL
from pathlib import Path

from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig, SampleEfficiency
from stackelberg_mbrl.envs.env_util import LearnableWorldModel, RandomMDP
from stackelberg_mbrl.envs.querying_env import ConstantContextEnv, CountedEnvWrapper, LeaderEnv, ModelQueryingEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

seeds = [23242, 234469, 3987, 411234, 1247]

def objective(trial: optuna.Trial):

    ## PPO params
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # Log scale for more precise tuning
    n_steps = trial.suggest_int('n_steps', 64, 2048)  # Reasonable range for PPO
    batch_size = trial.suggest_int('batch_size', 16, 256)  # Adjust batch size range
    n_epochs = trial.suggest_int('n_epochs', 1, 20)  # Number of times to reuse the data in the buffer
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)  # Clipping range for PPO
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)  # Entropy coefficient
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)  # Value function coefficient
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)  # Gradient clipping
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)  # GAE lambda
    # batch_size = trial.suggest_int('batch_size', 2, 100)
    # use_sde = trial.suggest_categorical('use_sde', [True, False])
    # ent_coef = trial.suggest_float('ent_coef', 0.0, 1.0)
    # vf_coef = trial.suggest_float('vf_coef', 0.0, 1.0)
    
    ppo_kwargs = {
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'max_grad_norm': max_grad_norm,
        'gae_lambda': gae_lambda,
        'learning_rate': learning_rate,
    }


    ## env
    env_reward_weight = trial.suggest_float('env_reward_weight', 0.0001, 1.0)
    # env_noise_weight = trial.suggest_float('env_noise_weight', 0.0, 1.0)

    def run(seed):
        config = ExperimentConfig(
                experiment_name='optuna_mal',
                env_config=EnvConfig(
                    env_true_id="simple_mdp_2",
                    env_eval_id="simple_mdp_2",
                    max_episode_steps=50
                ),
                policy_config=LoadPolicy(
                    path="stackelberg_mbrl/experiments/poster_mal_agent_reward/checkpoints/policy_simple_mdp_2.zip",
                ),
                leader_env_config=LeaderEnvConfig(
                    # env_noise_weight=lambda step: env_noise_weight,
                    env_reward_weight=env_reward_weight,
                    env_noise_weight=lambda step: 0.0,
                    # env_reward_weight=0.3,
                ),
                sample_efficiency=SampleEfficiency(
                    sample_eval_rate=80,
                    n_eval_episodes=10,
                    max_samples=15_000,
                ),
                world_model_config=WorldModelConfig(
                    total_training_steps=250_000,
                    ppo_kwargs=ppo_kwargs,
                ),
                seed=seed
            )
        try:
            return train_contextualized_MAL(config)
        except KeyboardInterrupt:
            raise KeyboardInterrupt()
        except:
            return None

    mean_rewards = []
    last_rewards = []

    for seed in seeds:
        data = run(seed)
        if data is None:    
            last_rewards.append(-100)
            mean_rewards.append(-100)
        else:
            last_rewards.append(data['reward'])
            mean_rewards.append(np.mean([x[1] for x in data['evals']]))

    mean_mean_r = np.mean(mean_rewards)
    mean_last_r = np.mean(last_rewards)
    ret = float(mean_mean_r + 10 * mean_last_r)
    print(f"mean mean reward: {mean_mean_r}, mean last reward: {mean_last_r}, score: {ret}")
    assert not np.isnan(ret) and not np.isinf(ret) and type(ret) == float
    return ret

###### 

# Create a study object and optimize the objective function
# storage_path = Path('stackelberg_mbrl') / 'experiments' / 'optuna_mal' / 'storage'
# storage = JournalFileStorage(str(storage_path))
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best set of hyperparameters
print('Best hyperparameters: ', study.best_params)
# Print the corresponding performance
print('Best performance: ', study.best_value)