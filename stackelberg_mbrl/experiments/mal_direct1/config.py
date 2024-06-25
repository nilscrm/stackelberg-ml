from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig, TableWorldModelConfig


poster_config = ExperimentConfig(
    experiment_name="mal_direct1",
    env_config=EnvConfig(
        env_true_id="simple_mdp_2",
        env_eval_id="simple_mdp_2",
        max_episode_steps=50
    ),
    # policy_config=PolicyConfig(
    #     pretrain_iterations=1,
    #     samples_per_training_iteration=1_000_000,
    #     model_save_name='policy_simple_mdp_2',
    # ),
    policy_config=LoadPolicy(
        path="stackelberg_mbrl/experiments/mal_direct1/checkpoints/policy_simple_mdp_2.zip",
    ),
    leader_env_config=LeaderEnvConfig(),
    world_model_config=TableWorldModelConfig(
        eps = 1e-8,
        noise = 0.3,
        batch_size = 1,
        max_training_samples = 50,
        init_sample_trajectories = 0,
    ),
    seed=14
)
