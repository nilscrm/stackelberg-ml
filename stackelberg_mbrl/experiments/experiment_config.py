from pathlib import Path
from typing import Callable, Dict, Any
from pydantic import BaseModel, FilePath, DirectoryPath, ConfigDict


class EnvConfig(BaseModel):
    env_true_id: str
    env_eval_id: str # handcrafted environment that the pretrained policy oracle can be tested on
    max_episode_steps: int = 50

class PolicyConfig(BaseModel):
    policy_kwargs: dict = dict(
        net_arch=dict(pi=[8, 8, 8, 8], qf=[40, 30]),
    )
    pretrain_iterations: int = 20
    samples_per_training_iteration: int = 10_000
    save_name: str | None = None

class LoadPolicy(BaseModel):
    path: FilePath

class LeaderEnvConfig(BaseModel):
    env_reward_weight: float = 0.0
    env_noise_weight: Callable[[int],float] = lambda x: 0.0 # given the current step, output the probability of a random trajectory

class WorldModelConfig(BaseModel):
    total_training_steps: int = 1_000_000
    save_name: str | None = None
    ppo_kwargs: Dict[str,Any] | None = None

class LoadWorldModel(BaseModel):
    path: FilePath

class SampleEfficiency(BaseModel):
    sample_eval_rate: int = 30 # how many samples inbetween two evaluations?
    n_eval_episodes: int = 15
    max_samples: int = 200
    log_save_name: str | None = None

class ExperimentConfig(BaseModel):
    # Make config immutable
    model_config = ConfigDict(frozen=True)

    experiment_name: str

    env_config: EnvConfig
    policy_config: PolicyConfig | LoadPolicy
    leader_env_config: LeaderEnvConfig
    world_model_config: WorldModelConfig | LoadWorldModel
    sample_efficiency: SampleEfficiency | None

    output_dir: DirectoryPath = Path("stackelberg_mbrl/experiments/")
    seed: int = 12

