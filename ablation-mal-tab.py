# seeds = [23242, 234469, 3987, 128, 411234]
seeds = [23242, 234469, 3987, 128]
# learning_rates = [1e-1, 9e-2, 6e-2, 3e-2, 1e-2, 1e-3, 9e-4, 6e-4, 3e-4, 1e-4]
# alpha_r = [0.0, 0.3, 0.6, 1.0]
alpha_n = [0.0, 0.25, 0.3, 0.5, 0.75, 1.0]

EXPERIMENT = "mal_tabular_noise_ablation"


################

from itertools import product
from stackelberg_mbrl.train_mal import train_contextualized_MAL
from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig, SampleEfficiency, TableWorldModelConfig
import numpy as np
from pathlib import Path

# {'init_sample_trajectories': 0, 'batch_size': 7, 'eps': 1.9908870331765223e-07, 'default_model': 0.08329231349702726, 'env_noise_weight': 0.9153182656744037}

# for lr in 2**np.linspace(np.log2(1e-5), np.log2(0.3), 4):
# for an in alpha_n:
for seed in seeds:
    sample_name = f"seed_{seed}"
    p = Path('stackelberg_mbrl') / 'experiments' / EXPERIMENT / "sample_efficiency" / sample_name
    if p.exists(): 
        print(f"Skipping {sample_name}")
        continue
    else:
        print(f"Training {sample_name}")
    try:
        train_contextualized_MAL(ExperimentConfig(
                experiment_name=EXPERIMENT,
                env_config=EnvConfig(
                    env_true_id="simple_mdp_2",
                    env_eval_id="simple_mdp_2",
                    max_episode_steps=50
                ),
                policy_config=LoadPolicy(
                    path="stackelberg_mbrl/experiments/poster_mal_agent_reward/checkpoints/policy_simple_mdp_2.zip",
                ),
                leader_env_config=LeaderEnvConfig(),
                sample_efficiency=SampleEfficiency(
                    sample_eval_rate=10,
                    n_eval_episodes=15,
                    log_save_name=sample_name,
                    max_samples=0,
                ),
                world_model_config=TableWorldModelConfig(
                    max_training_samples=300,
                    init_sample_trajectories=0,
                    noise=0.9256869566965958,
                    batch_size=1,
                    # default_model=0.08329231349702726,
                    # eps=1.9908870331765223e-07
                ),
                seed=seed
            ), verbose=True)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    # except:
    #     print(f'Skipping {sample_name} due to error')