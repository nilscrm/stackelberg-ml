from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig


model_rl_config = ExperimentConfig(
    experiment_name="model_rl",
    env_config=EnvConfig(
        env_true_id="simple_mdp_2_variant_2",
        env_eval_id="simple_mdp_2_variant_1",
        max_episode_steps=50
    ),
    policy_config=LoadPolicy(
        path="stackelberg_mbrl/experiments/model_rl/checkpoints/policy.zip",
    ),
    leader_env_config=LeaderEnvConfig(
        env_reward_weight=1.0
    ),
    world_model_config=WorldModelConfig(
        model_save_name="model",
    ),
)
