from tqdm import tqdm
from envs.learned_env import DiscreteLearnedEnv
from envs.simple_mdp import SimpleMDPEnv
from nn.model.world_models import WorldModel, RandomDiscreteModel
from algos.model_based_npg import ModelBasedNPG
from nn.policy.policy_networks import PolicyMLP
from nn.baseline.baselines import BaselineMLP
from policies.contextualized_policy import ContextualizedPolicy
import torch
import torch.nn.functional as F
import numpy as np

from nn.policy.stable_baseline_policy_networks import SB3ContextualizedFeatureExtractor
from policies.stable_baseline_policy import SB3DiscretePolicy
from util.tensor_util import one_hot
from util.trajectories import sample_trajectories

from itertools import product

from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def train_contextualized_MAL():
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
        "seed": 1234,
        "policy_size": (32, 32),
        "device": "cpu",
        "npg_step_size": 0.05,
        "training_iterations": 1000,
        "init_samples": 500,
        "policy_pretrain_steps": 200,# TODO: make this sth large, like 1000,
        "policy_inner_training_steps": 3,
        "model_batch_size": 64,
        "model_fit_epochs": 5, # TODO: should this be 1 since we essentially want best-response, technically, as soon as we do one gradient step, the trajectories are no longer best-response
        "policy_trajectories_per_step": 250,
        "max_episode_steps": 50,
        "num_models": 4,
        "learn_reward": False,
    }

    np.random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])

    # Groundtruth environment, which we sample from
    env_true = SimpleMDPEnv(max_episode_steps=config["max_episode_steps"])

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
    # TODO: figure out how to sample random rewards...
    random_model = RandomDiscreteModel(env_true, init_state_probs, termination_func, reward_func, min_reward=-5, max_reward=100)
    random_env = DiscreteLearnedEnv(random_model, env_true.action_space, env_true.observation_space, config["max_episode_steps"])

    # context = in which state will we land (+ what reward we get) for each query
    observation_space = F.one_hot(torch.arange(env_true.observation_dim, requires_grad=False), num_classes=env_true.observation_dim).float()
    action_space = F.one_hot(torch.arange(env_true.action_dim, requires_grad=False), num_classes=env_true.action_dim).float()

    dynamics_queries = list(product(observation_space, action_space))
    reward_queries = list(product(observation_space, action_space, observation_space))
    context_size = len(dynamics_queries) * env_true.observation_dim + len(reward_queries)

    policy_kwargs = dict(
        features_extractor_class=SB3ContextualizedFeatureExtractor,
        features_extractor_kwargs=dict(context_size=context_size),
    )
    trainer = PPO("MlpPolicy", random_env, policy_kwargs=policy_kwargs)
    
    # Pretrain the policy conditioned on a world model
    print("Pretraining")
    for iter in range(config["policy_pretrain_steps"]):
        # create "random" world model = basically random transition probabilities (and random reward if learned)
        random_model.randomize()
        trainer.policy.features_extractor.set_context(random_model.query(dynamics_queries, reward_queries))

        for i in range(config["policy_inner_training_steps"]):
            trainer.learn(config["policy_trajectories_per_step"])

        print(evaluate_policy(trainer.policy, random_env, n_eval_episodes=10))

        # TODO: how do we know we have converged? => we should do some sort of validation to see if we are still improving

    # Train model (leader)
    # NOTE: We are not using a replay buffer because then some trajectories are produced by best-response policies to older world models, violating the follower-best-reponse criteria.
    #       Even though we are only using the trajectories to learn to predict the next states given a state-action-pair, having a non-best-response state-visitation distribution, will skew the weighting in the loss, giving us a sub-optimal policy-model combination.
    # TODO: on the other hand, this means we are probably less sample efficient, so it might be worth to try both (probably it converges in the end because the world model wont change much and so the policy will be consistent and thus also the trajectories)
    policy_oracle = SB3DiscretePolicy(trainer.policy)
    print("Training")
    for iter in range(config["training_iterations"]):
        print(f"Training iteration {iter}")

        # Sample trajectories on the environment, using the best-response-policy (wrt the current model)
        policy_oracle.policy.features_extractor.set_context(model.query(dynamics_queries, reward_queries))
        # TODO: these trajectories need to contain the context, so the leader sees that it is being queried (look at how gerstgrasser did it)
        # we could just start with samples that are from the queries
        env_trajectories = sample_trajectories(env_true, policy_oracle, max_steps=config["max_episode_steps"], num_trajectories=config["init_samples"]) 

        states = np.concatenate(env_trajectories.states)
        actions = np.concatenate(env_trajectories.actions)
        rewards = np.concatenate(env_trajectories.rewards)
        next_states = np.concatenate(env_trajectories.next_states)

        average_reward = np.mean([trajectory.total_reward for trajectory in env_trajectories])
        print(f"\tAverage reward: {average_reward:.3f}")
        
        dynamics_loss = model.fit_dynamics(states, actions, next_states, 
                                           fit_epochs=config["model_fit_epochs"], fit_mb_size=config["model_batch_size"])
        print(f"\tDynamics loss: {np.mean(dynamics_loss):.3f}")

        if config["learn_reward"]:
            reward_loss = model.fit_reward(states, actions, rewards, 
                                           fit_epochs=config["model_fit_epochs"], fit_mb_size=config["model_batch_size"])
            print(f"\tReward loss: {np.mean(reward_loss):.3f}")
        
        

if __name__ == '__main__':
    train_contextualized_MAL()
