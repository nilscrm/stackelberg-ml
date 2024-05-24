from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig


poster_config = ExperimentConfig(
    experiment_name="poster_pal",
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
        env_reward_weight=0.0
    ),
    world_model_config=WorldModelConfig(
        total_training_steps=250_000,
        model_save_name="simple_mdp_2",
    ),
    seed=14
)
