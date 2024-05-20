from itertools import product
import numpy as np
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import gymnasium

import stackelberg_mbrl.envs.simple_mdp
from stackelberg_mbrl.envs.querying_env import LeaderEnv, ModelQueryingEnv
from stackelberg_mbrl.envs.env_util import transition_probabilities_from_world_model, draw_mdp, RandomMDP, LearnableWorldModel
from stackelberg_mbrl.experiments.experiment_config import ExperimentConfig, LoadPolicy, PolicyConfig, LoadWorldModel, WorldModelConfig
from stackelberg_mbrl.experiments.model_rl.config import model_rl_config


def train_contextualized_MAL(config: ExperimentConfig):
    """
    In contextualized MAL we condition and pretrain the policy on random models.
    This way we get an oracle that behaves like the best policy conditioned on each model.
    In doing so, we can then optimize for an optimal model-policy pair.

    Hypothesis:
    - helpful in environments where small changes in the model drastically change the best policy
    - requires less samples from the env because we train in hypothetical environments (=> can trade expensive sampling for compute)
        (actually, this is not true, because training a best response policy in an inner loop wouldn't incur any additional samples from the environment)
    """
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    # Groundtruth environment, which we sample from
    env_true = gymnasium.make(config.env_config.env_true_id, max_ep_steps=config.env_config.max_episode_steps)
    queries = list(product(range(env_true.num_states), range(env_true.num_actions)))
    querying_env_true = ModelQueryingEnv(env_true, queries)
    env_eval = gymnasium.make(config.env_config.env_eval_id, max_ep_steps=config.env_config.max_episode_steps)
    querying_env_eval = ModelQueryingEnv(env_eval, queries)

    env_true.draw_mdp(config.output_dir / config.experiment_name / "mdps" / "env_true.png")
    env_eval.draw_mdp(config.output_dir / config.experiment_name / "mdps" / "env_variant.png")

    random_mdp = RandomMDP(
        num_states=env_true.num_states,
        num_actions=env_true.num_actions,
        rewards=env_true.rewards,
        initial_state=env_true.initial_state,
        final_state=env_true.final_state,
        max_ep_steps=env_true.max_ep_steps,
    )
    querying_random_mdp = ModelQueryingEnv(random_mdp, queries)

    eval_envs = [
        ModelQueryingEnv(
            RandomMDP(
                num_states=env_true.num_states,
                num_actions=env_true.num_actions,
                rewards=env_true.rewards,
                initial_state=env_true.initial_state,
                final_state=env_true.final_state,
                max_ep_steps=env_true.max_ep_steps,
                randomize_on_reset=False,
            ),
            queries,
        )
        for _ in range(10)
    ]

    for i, eval_env in enumerate(eval_envs):
        eval_env.world_model.draw_mdp(config.output_dir / config.experiment_name / f"mdps/eval_mdp_{i}.png")

    match config.policy_config:
        case LoadPolicy():
            print("Loading policy model from file.")
            policy_ppo = PPO.load(config.policy_config.path, querying_random_mdp)
        case PolicyConfig():
            policy_config: PolicyConfig = config.policy_config
            policy_ppo = PPO(
                "MlpPolicy",
                querying_random_mdp,
                policy_kwargs=policy_config.policy_kwargs,
                tensorboard_log=config.output_dir / config.experiment_name / "tb",
            )

            # Pretrain the policy conditioned on a world model
            print("Pretraining policy model")
            for iter in range(policy_config.pretrain_iterations):
                policy_ppo.learn(policy_config.samples_per_training_iteration, tb_log_name="Policy", reset_num_timesteps=False)

                print(f"Pretraining Iteration {iter}")
                with torch.no_grad():
                    for i, eval_random_env in enumerate(eval_envs):
                        eval_random_mean, eval_random_std = evaluate_policy(policy_ppo.policy, eval_random_env, n_eval_episodes=5)
                        print(f"\tAvg Reward (random model {i}):      {eval_random_mean:.3f} ± {eval_random_std:.3f}")

                    eval_true_mean, eval_true_std = evaluate_policy(policy_ppo.policy, querying_env_true, n_eval_episodes=10)
                    print(f"\tAvg Reward (true env):      {eval_true_mean:.3f} ± {eval_true_std:.3f}")

                    eval_variant_mean, eval_variant_std = evaluate_policy(policy_ppo.policy, querying_env_eval, n_eval_episodes=10)
                    print(f"\tAvg Reward (eval env):   {eval_variant_mean:.3f} ± {eval_variant_std:.3f}")

                # TODO: how do we know we have converged? => we should do some sort of validation to see if we are still improving

            if policy_config.model_save_name is not None:
                policy_ppo.save(config.output_dir / config.experiment_name / "checkpoints" / policy_config.model_save_name)

    leader_env = LeaderEnv(env_true, policy_ppo.policy, queries, reward=config.world_model_config.leader_reward)

    match config.world_model_config:
        case LoadWorldModel():
            model_ppo = PPO.load(config.world_model_config.path, leader_env)
        case WorldModelConfig():
            model_config: WorldModelConfig = config.world_model_config
            model_ppo = PPO(
                "MlpPolicy",
                leader_env,
                tensorboard_log=config.output_dir / config.experiment_name / "tb",
                gamma=1.0,
                # use_sde=True,
            )

            draw_mdp(
                transition_probabilities_from_world_model(model_ppo.policy, env_true.num_states, env_true.num_actions),
                env_true.rewards,
                config.output_dir / config.experiment_name / "mdps" / "initial_model.png",
            )

            print("Training world model")
            model_ppo.learn(total_timesteps=model_config.total_training_steps, progress_bar=True, tb_log_name="WorldModel")

            if model_config.model_save_name is not None:
                model_ppo.save(config.output_dir / config.experiment_name / "checkpoints" / model_config.model_save_name)

    # Evaluation of model
    print(f"Model reward: {evaluate_policy(model_ppo.policy, leader_env)}")

    model_querying_env = ModelQueryingEnv(
        LearnableWorldModel(
            model_ppo.policy,
            env_true.num_states,
            env_true.num_actions,
            env_true.max_ep_steps,
            env_true.rewards,
            env_true.initial_state,
            env_true.final_state,
        ),
        queries,
    )
    with torch.no_grad():
        policy_reward, policy_reward_std = evaluate_policy(policy_ppo.policy, model_querying_env, n_eval_episodes=10)
        print(f"Avg Policy Reward on learned model:   {policy_reward:.3f} ± {policy_reward_std:.3f}")

    draw_mdp(
        transition_probabilities_from_world_model(model_ppo.policy, env_true.num_states, env_true.num_actions),
        env_true.rewards,
        config.output_dir / config.experiment_name / "mdps" / "final_model.png",
    )


if __name__ == "__main__":
    train_contextualized_MAL(model_rl_config)
