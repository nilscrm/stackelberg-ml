seeds = [23242, 234469, 3987, 128, 411234]
experiment = 'pal_sample_efficiency'

################

from stackelberg_mbrl.train_pal import train_contextualized_PAL
from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig, SampleEfficiency
import numpy as np
from pathlib import Path


for seed in seeds:
    sample_name = f"seed_{seed}"
    p = Path('stackelberg_mbrl') / 'experiments' / experiment / "sample_efficiency" / sample_name
    if p.exists(): 
        print(f"Skipping {sample_name}")
        continue
    else:
        print(f"Training {sample_name}")
    try:
        pass
        config = ExperimentConfig(
                experiment_name=experiment,
                env_config=EnvConfig(
                    env_true_id="simple_mdp_2",
                    env_eval_id="simple_mdp_2",
                    max_episode_steps=50
                ),
                policy_config=PolicyConfig(
                    pretrain_iterations=1,
                    samples_per_training_iteration=250_000,
                ),
                leader_env_config=LeaderEnvConfig(
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
            )
        train_contextualized_PAL(config, verbose=True)
    except:
        print(f'Skipping {sample_name} due to error')