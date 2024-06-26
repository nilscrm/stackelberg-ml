seeds = [23242, 234469, 3987, 128, 411234]
learning_rates = [1e-1, 9e-2, 6e-2, 3e-2, 1e-2, 1e-3, 9e-4, 6e-4, 3e-4, 1e-4]



################

from stackelberg_mbrl.train_mal import train_contextualized_MAL
from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig, SampleEfficiency
import numpy as np
from pathlib import Path

# for lr in 2**np.linspace(np.log2(1e-5), np.log2(0.3), 4):
for lr in learning_rates:
    for seed in seeds:
        sample_name = f"lr_{lr}_seed_{seed}"
        p = Path('stackelberg_mbrl') / 'experiments' / 'mal_lr_ablation' / "sample_efficiency" / sample_name
        if p.exists(): 
            print(f"Skipping {sample_name}")
            continue
        else:
            print(f"Training {sample_name}")
        try:
            train_contextualized_MAL(ExperimentConfig(
                    experiment_name='mal_lr_ablation',
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
                        # env_reward_weight=env_reward_weight,
                        env_noise_weight=lambda step: 0.0,
                        env_reward_weight=0.3,
                        learning_rate=lr,
                        n_steps=100,
                    ),
                    sample_efficiency=SampleEfficiency(
                        sample_eval_rate=50,
                        n_eval_episodes=15,
                        max_samples=20_000,
                        log_save_name=sample_name
                    ),
                    world_model_config=WorldModelConfig(
                        total_training_steps=250_000,
                    ),
                    seed=seed
                ), verbose=True)
        except:
            print(f'Skipping {sample_name} due to error')