# seeds = [23242, 234469, 3987, 128, 411234]
seeds = [23242, 234469, 3987, 128]
# learning_rates = [1e-1, 9e-2, 6e-2, 3e-2, 1e-2, 1e-3, 9e-4, 6e-4, 3e-4, 1e-4]
alpha_r = [0.0, 0.3, 0.6, 1.0]
alpha_n = [0.0, 0.3, 0.75, 1.0]

EXPERIMENT = "mal_alpha_reward_ablation_hyper"


################

from itertools import product
from stackelberg_mbrl.train_mal import train_contextualized_MAL
from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig, SampleEfficiency
import numpy as np
from pathlib import Path

# for lr in 2**np.linspace(np.log2(1e-5), np.log2(0.3), 4):
for ar, an in product(alpha_r, alpha_n):
    for seed in seeds:
        sample_name = f"reward_{ar}_noise_{an}_seed_{seed}"
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
                    leader_env_config=LeaderEnvConfig(
                        env_noise_weight=lambda step: an,
                        env_reward_weight=ar,
                    ),
                    sample_efficiency=SampleEfficiency(
                        sample_eval_rate=100,
                        n_eval_episodes=15,
                        max_samples=15_000,
                        log_save_name=sample_name
                    ),
                    world_model_config=WorldModelConfig(
                        total_training_steps=250_000,
                        ppo_kwargs={'learning_rate': 0.006128518911011046, 'n_steps': 188, 'batch_size': 195, 'n_epochs': 4, 'clip_range': 0.3802060420065968, 'ent_coef': 0.0045823258311440025, 'vf_coef': 0.9032640940522102, 'max_grad_norm': 0.6807346408015829, 'gae_lambda': 0.9915784607770367}
                    ),
                    seed=seed
                ), verbose=True)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print(f'Skipping {sample_name} due to error')