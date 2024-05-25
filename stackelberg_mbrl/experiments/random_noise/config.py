from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy


random_noise_config = ExperimentConfig(
    experiment_name="random_noise",
    env_config=EnvConfig(
        # env_true_id="simple_mdp_2_variant_2",
        env_true_id="simple_mdp_2",
        env_eval_id="simple_mdp_2_variant_1",
        max_episode_steps=50
    ),
    policy_config=LoadPolicy(
        path="stackelberg_mbrl/experiments/random_noise/checkpoints/policy.zip",
    ),
    world_model_config=WorldModelConfig(
        model_save_name="model",
    ),
)
