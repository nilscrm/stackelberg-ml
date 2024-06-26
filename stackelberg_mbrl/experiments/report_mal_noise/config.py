from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, EnvConfig, PolicyConfig, WorldModelConfig, LoadPolicy, LeaderEnvConfig, SampleEfficiency


poster_config = ExperimentConfig(
    experiment_name="report_mal_noise",
    env_config=EnvConfig(
        env_true_id="simple_mdp_2",
        env_eval_id="simple_mdp_2",
        max_episode_steps=50
    ),
    policy_config=LoadPolicy(
        # just use the same one....
        path="stackelberg_mbrl/experiments/poster_mal_agent_reward/checkpoints/policy_simple_mdp_2.zip",
    ),
    # policy_config=PolicyConfig(
    #     pretrain_iterations=1,
    #     samples_per_training_iteration=1_000_000,
    #     model_save_name="policy_simple_mdp_2",
    # ),
    leader_env_config=LeaderEnvConfig(
        env_noise_weight=lambda step: 1.0
    ),
    # sample_efficiency=None,
    sample_efficiency=SampleEfficiency(
        sample_eval_rate=100,
        n_eval_episodes=30,
        max_samples=20_000,
        log_save_name="alpha_1.0"
    ),
    world_model_config=WorldModelConfig(
        total_training_steps=250_000,
        # total_training_steps=1_000,
        # model_save_name="simple_mdp_2",
    ),
    seed=14
)
