from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy


model_rl_config = ExperimentConfig(
    experiment_name="model_rl",
    env_config=EnvConfig(),
    policy_config=LoadPolicy(
        path="stackelberg_mbrl/experiments/model_rl/checkpoints/policy.zip",
    ),
    world_model_config=WorldModelConfig(
        model_save_name="model",
    ),
)
