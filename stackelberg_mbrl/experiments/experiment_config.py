from pathlib import Path
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
    model_save_name: str | None = None

class LoadPolicy(BaseModel):
    path: FilePath

class WorldModelConfig(BaseModel):
    total_training_steps: int = 1_000_000
    model_save_name: str | None = None

class LoadWorldModel(BaseModel):
    path: FilePath

class ExperimentConfig(BaseModel):
    # Make config immutable
    model_config = ConfigDict(frozen=True)

    experiment_name: str

    env_config: EnvConfig
    policy_config: PolicyConfig | LoadPolicy
    world_model_config: WorldModelConfig | LoadWorldModel

    output_dir: DirectoryPath = Path("stackelberg_mbrl/experiments/")
    seed: int = 12
