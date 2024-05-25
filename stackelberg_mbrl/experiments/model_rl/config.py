from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig


model_rl_config = ExperimentConfig(
    experiment_name="model_rl",
    env_config=EnvConfig(
        env_true_id="ergodic_mdp_1",
        env_eval_id="ergodic_mdp_1",
        max_episode_steps=50
    ),
    policy_config=PolicyConfig(),
    leader_env_config=LeaderEnvConfig(
        env_reward_weight=0.0
    ),
    world_model_config=WorldModelConfig(
        total_training_steps=250_000,
        model_save_name="ergodic_mdp_1",
    ),
    seed=14
)
