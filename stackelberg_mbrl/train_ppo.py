import torch
import torch.nn.functional as F
import numpy as np

from stackelberg_mbrl.envs.learned_env import DiscreteLearnedEnv
from stackelberg_mbrl.envs.querying_env import LeaderEnv
from stackelberg_mbrl.envs.simple_mdp import *
from stackelberg_mbrl.envs.env_util import transition_probabilities_from_world_model, draw_mdp
from stackelberg_mbrl.nn.model.world_models import WorldModel, StaticDiscreteModel
from stackelberg_mbrl.nn.policy.stable_baseline_policy_networks import SB3ContextualizedFeatureExtractor
from stackelberg_mbrl.policies.stable_baseline_policy import SB3DiscretePolicy
from stackelberg_mbrl.util.tensor_util import one_hot
from stackelberg_mbrl.util.trajectories import sample_trajectories

from itertools import product

from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def train_contextualized_MAL(experiment_name: str = "ergodic_1"):
    """
        In contextualized MAL we condition and pretrain the policy on random models. 
        This way we get an oracle that behaves like the best policy conditioned on each model.
        In doing so, we can then optimize for an optimal model-policy pair.

        Hypothesis:
        - helpful in environments where small changes in the model drastically change the best policy
        - requires less samples from the env because we train in hypothetical environments (=> can trade expensive sampling for compute)
            (actually, this is not true, because training a best response policy in an inner loop wouldn't incur any additional samples from the environment)
    """

    config = {
        "seed": 12,
        "policy_size": (32, 32),
        "device": "cpu",
        "npg_step_size": 0.05,
        "training_iterations": 1000,
        "init_samples": 500,
        "policy_pretrain_steps": 0,# TODO: make this sth large, like 1000,
        "model_training_steps": 1_000_000,
        "policy_inner_training_steps": 1,
        "model_batch_size": 64,
        "model_fit_epochs": 5, # TODO: should this be 1 since we essentially want best-response, technically, as soon as we do one gradient step, the trajectories are no longer best-response
        "policy_trajectories_per_step": 250,
        "max_episode_steps": 50,
        "num_models": 4,
        "learn_reward": False,
        # Set to None for no loading
        "load_pretrained_policy_file": None, #f"stackelberg_mbrl/experiments/{experiment_name}/checkpoints/pretrained_policy",
        "pretrained_policy_save_file": None,
        "model_save_file": f"stackelberg_mbrl/experiments/{experiment_name}/train_model/checkpoints/model",
    }

    np.random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])

    # Groundtruth environment, which we sample from
    env_true = ergodic_mdp_1(max_episode_steps=config["max_episode_steps"])
    env_variant = simple_mdp_2_variant(max_episode_steps=config["max_episode_steps"])

    env_true.draw_mdp(f"stackelberg_mbrl/experiments/{experiment_name}/mdps/env_true.png")
    env_variant.draw_mdp(f"stackelberg_mbrl/experiments/{experiment_name}/mdps/env_variant.png")

    reward_func = None
    if not config["learn_reward"]:
        reward_func = getattr(env_true, "reward", None)

    termination_func = getattr(env_true, "is_done", None)

    # Sample true initial state distribution
    init_state_probs = []
    for i in range(100):
        init_state,_ = env_true.reset()
        init_state_probs.append(one_hot(init_state, env_true.observation_dim).numpy())
    init_state_probs = np.mean(init_state_probs, axis=0)

    # NOTE: in this scenario it does not make sense to have multiple world models, as they would all converge to a stackelberg equilibrium and not help to find the best policy
    model = WorldModel(state_dim=env_true.observation_dim, act_dim=env_true.action_dim, hidden_sizes=(64,64), reward_func=reward_func)
    model.draw_mdp(f"stackelberg_mbrl/experiments/{experiment_name}/mdps/model/0.png")
    # TODO: figure out how to sample random rewards...
    random_model = StaticDiscreteModel(env_true, init_state_probs, termination_func, reward_func)
    random_model_env = DiscreteLearnedEnv(random_model, env_true.action_space, env_true.observation_space, config["max_episode_steps"])

    random_model.draw_mdp(f"stackelberg_mbrl/experiments/{experiment_name}/mdps/random_model.png")

    # context = in which state will we land (+ what reward we get) for each query
    observation_space = F.one_hot(torch.arange(env_true.observation_dim, requires_grad=False), num_classes=env_true.observation_dim).float()
    action_space = F.one_hot(torch.arange(env_true.action_dim, requires_grad=False), num_classes=env_true.action_dim).float()

    dynamics_queries = list(product(observation_space, action_space))
    reward_queries = list(product(observation_space, action_space, observation_space)) if config["learn_reward"] else []
    context_size = len(dynamics_queries) * env_true.observation_dim + len(reward_queries)

    if config["load_pretrained_policy_file"] is None:
        policy_kwargs = dict(
            features_extractor_class=SB3ContextualizedFeatureExtractor,
            features_extractor_kwargs=dict(context_size=context_size),
            net_arch=dict(pi=[8,8,8,8], qf=[40,30])
        )
        # Note: It's importatnt that we reduce the number of steps to a small number so that we randomize the environment
        # frequenctly and don't overfit to individual world models
        trainer = PPO("MlpPolicy", random_model_env, policy_kwargs=policy_kwargs, n_epochs=1, n_steps=10)
    else:
        print("Loading policy model from file.")
        trainer = PPO.load(config["load_pretrained_policy_file"], random_model_env)
    
    # Prepare Evaluation
    # sample a few fixed random models which we will evaluate on (achievable rewards are different between models!)
    eval_random_models = [
        DiscreteLearnedEnv(StaticDiscreteModel(env_true, init_state_probs, termination_func, reward_func), 
                           env_true.action_space, env_true.observation_space, max_episode_steps=config["max_episode_steps"])
        for i in range(1)]
    
    for i, eval_model in enumerate(eval_random_models):
        eval_model.env_model.draw_mdp(f"stackelberg_mbrl/experiments/{experiment_name}/mdps/eval_models/eval_model_{i}.png")

    # check how good we are currently doing on the best possible environment (the true one)
    eval_true_env = DiscreteLearnedEnv(StaticDiscreteModel(env_true, init_state_probs, termination_func, reward_func), 
                           env_true.action_space, env_true.observation_space, max_episode_steps=config["max_episode_steps"])
    eval_true_env.env_model.set_transition_probs(torch.from_numpy(np.array(env_true.transitions)).float())
    env_true_queries = eval_true_env.env_model.query(dynamics_queries, reward_queries).detach()

    # check how good we are currently doing on another fixed environment
    eval_variant_env = DiscreteLearnedEnv(StaticDiscreteModel(env_variant, init_state_probs, termination_func, reward_func), 
                           env_true.action_space, env_true.observation_space, max_episode_steps=config["max_episode_steps"])
    eval_variant_env.env_model.set_transition_probs(torch.from_numpy(np.array(env_variant.transitions)).float())

    # Pretrain the policy conditioned on a world model
    print("Pretraining")
    for iter in range(config["policy_pretrain_steps"]):
        # create "random" world model = basically random transition probabilities (and random reward if learned)
        random_model.randomize()
        trainer.policy.features_extractor.set_context(random_model.query(dynamics_queries, reward_queries))

        for i in range(config["policy_inner_training_steps"]):
            trainer.learn(config["policy_trajectories_per_step"])

        # Eval
        if iter % 100 == 0:
            print(f"Pretraining Iteration {iter}")
            with torch.no_grad():
                eval_rand_means = []
                # for eval_random_model_env in eval_random_models:
                #     trainer.policy.features_extractor.set_context(eval_random_model_env.env_model.query(dynamics_queries, reward_queries))  
                #     eval_rand_mean, eval_rand_std = evaluate_policy(trainer.policy, eval_random_model_env, n_eval_episodes=5)
                #     eval_rand_means.append(eval_rand_mean)
                
                # print(f"\tAvg Reward (random models): {np.mean(eval_rand_mean):.3f}")

                trainer.policy.features_extractor.set_context(eval_true_env.env_model.query(dynamics_queries, reward_queries))  
                eval_true_mean,eval_true_std = evaluate_policy(trainer.policy, env_true, n_eval_episodes=10)
                print(f"\tAvg Reward (true env):      {eval_true_mean:.3f} ± {eval_true_std:.3f}")

                trainer.policy.features_extractor.set_context(eval_variant_env.env_model.query(dynamics_queries, reward_queries))  
                eval_variant_mean,eval_variant_std = evaluate_policy(trainer.policy, env_true, n_eval_episodes=10)
                print(f"\tAvg Reward (variant env):   {eval_variant_mean:.3f} ± {eval_variant_std:.3f}")

        # TODO: how do we know we have converged? => we should do some sort of validation to see if we are still improving

    if config["pretrained_policy_save_file"] is not None:
        trainer.save(path=config["pretrained_policy_save_file"])

    # Train model (leader)
    # NOTE: We are not using a replay buffer because then some trajectories are produced by best-response policies to older world models, violating the follower-best-reponse criteria.
    #       Even though we are only using the trajectories to learn to predict the next states given a state-action-pair, having a non-best-response state-visitation distribution, will skew the weighting in the loss, giving us a sub-optimal policy-model combination.
    # TODO: on the other hand, this means we are probably less sample efficient, so it might be worth to try both (probably it converges in the end because the world model wont change much and so the policy will be consistent and thus also the trajectories)
    policy_oracle = SB3DiscretePolicy(trainer.policy)

    dynamics_queries = list(product(range(env_true.observation_dim), range(env_true.action_dim)))
    leader_env = LeaderEnv(env_true, trainer.policy, dynamics_queries)

    model_ppo = PPO("MlpPolicy", leader_env, tensorboard_log="stackelberg_mbrl/experiments/train_model/tb", gamma=0.99, use_sde=True)

    draw_mdp(
        transition_probabilities_from_world_model(model_ppo.policy, env_true.observation_dim, env_true.action_dim),
        env_true.rewards,
        f"stackelberg_mbrl/experiments/{experiment_name}/mdps/initial_model.png"
    )

    print("Training model")
    model_ppo.learn(total_timesteps=config["model_training_steps"], progress_bar=True)

    print(f"Model reward: {evaluate_policy(model_ppo.policy, leader_env)}")
    print(f"True Env reward: {evaluate_policy(model_ppo.policy, env_true)}")

    if config["model_save_file"] is not None:
        model_ppo.save(config["model_save_file"])

    draw_mdp(
        transition_probabilities_from_world_model(model_ppo.policy, env_true.observation_dim, env_true.action_dim),
        env_true.rewards,
        f"stackelberg_mbrl/experiments/{experiment_name}/mdps/final_model.png"
    )

    # for iter in range(config["training_iterations"]):
    #     print(f"Training iteration {iter}")

    #     # Sample trajectories on the environment, using the best-response-policy (wrt the current model)
    #     policy_oracle.policy.features_extractor.set_context(model.query(dynamics_queries, reward_queries))
    #     # TODO: these trajectories need to contain the context, so the leader sees that it is being queried (look at how gerstgrasser did it)
    #     # we could just start with samples that are from the queries
    #     env_trajectories = sample_trajectories(env_true, policy_oracle, max_steps=config["max_episode_steps"], num_trajectories=config["init_samples"]) 

    #     states = np.concatenate(env_trajectories.states)
    #     actions = np.concatenate(env_trajectories.actions)
    #     rewards = np.concatenate(env_trajectories.rewards)
    #     next_states = np.concatenate(env_trajectories.next_states)

    #     average_reward = np.mean(env_trajectories.total_rewards)

    #     if iter % 25 == 0:
    #         print(f"\tAverage reward: {average_reward:.3f}")
            
    #         dynamics_loss = model.fit_dynamics(states, actions, next_states, 
    #                                         fit_epochs=config["model_fit_epochs"], fit_mb_size=config["model_batch_size"])
    #         print(f"\tDynamics loss: {np.mean(dynamics_loss):.3f}")
    #         print(f"\tMSE Queries (true): {torch.mean((model.query(dynamics_queries, reward_queries) - env_true_queries)**2).item()}")

    #         if config["learn_reward"]:
    #             reward_loss = model.fit_reward(states, actions, rewards, 
    #                                         fit_epochs=config["model_fit_epochs"], fit_mb_size=config["model_batch_size"])
    #             print(f"\tReward loss: {np.mean(reward_loss):.3f}")
            
    #         print(f"\tState Visitation Frequency: {np.mean(states, axis=0)}")
    #         print(f'\tSample Trajectory: {env_trajectories[0].to_string(["A", "B", "C"], ["X", "Y"])}')

    # model.draw_mdp(f"stackelberg_mbrl/experiments/{experiment_name}/mdps/model/final.png")


if __name__ == '__main__':
    train_contextualized_MAL()
