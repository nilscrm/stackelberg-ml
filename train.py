from envs.gym_env import GymEnv
from envs.learned_env import LearnedEnv
from nn.model.world_models import WorldModel, RandomDiscreteModel
from algos.model_based_npg import ModelBasedNPG
from nn.policy.policy_networks import PolicyMLP
from nn.baseline.baselines import BaselineMLP, AverageBaseline
from policies.contextualized_policy import ModelContextualizedPolicy
import torch
import numpy as np

from util.trajectories import sample_trajectories

from itertools import product


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
        "seed": 123,
        "policy_size": (32, 32),
        "init_log_std": -0.5,
        "min_log_std": -2.5,
        "device": "cpu",
        "npg_step_size": 0.05,
        "training_iterations": 1000,
        "init_samples": 500,
        "policy_inner_training_steps": 5,
        "policy_pretrain_steps": 1000,
        "model_batch_size": 64,
        "model_fit_epochs": 10,
        "policy_trajectories_per_step": 250,
        "max_steps": 50,
        "num_models": 4,
        "learn_reward": True,
    }

    np.random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])

    # Groundtruth environment, which we sample from
    env_true = GymEnv("state-machine", act_repeat=1)
    env_true.set_seed(config["seed"])

    # NOTE: in this scenario it does not make sense to have multiple world models, as they would all converge to a stackelberg equilibrium and not help to find the best policy
    model = WorldModel(state_dim=env_true.observation_dim, act_dim=env_true.action_dim, learn_reward=config["learn_reward"])
    # TODO: figure out how to sample random rewards...
    random_model = RandomDiscreteModel(state_dim=env_true.observation_dim, act_dim=env_true.action_dim, min_reward=-5, max_reward=100)

    # context = in which state will we land (+ what reward we get) for each query
    dynamics_queries = product(range(env_true.observation_dim), range(env_true.action_dim))
    rewards_queries = product(range(env_true.observation_dim), range(env_true.action_dim), range(env_true.observation_dim))
    context_size = len(dynamics_queries) * env_true.observation_dim + len(rewards_queries)
    # NOTE: changed the policy model from gaussian MLP to one that predicts a distribution over actions (makes more sense for discrete action spaces + running log_std_dev makes no sense if we change the world model all the time)
    policy = PolicyMLP(env_true.observation_dim, env_true.action_dim, hidden_sizes=config['policy_size'], context_size=context_size)
    contextualized_policy = ModelContextualizedPolicy(policy, dynamics_queries)

    # baseline = BaselineMLP(input_dim=env_true.observation_dim, reg_coef=1e-3, batch_size=128, epochs=1,  learn_rate=1e-3)
    baseline = AverageBaseline()

    trainer = ModelBasedNPG(policy=contextualized_policy, normalized_step_size=config['npg_step_size'], save_logs=True)
    
    # Pretrain the policy conditioned on a world model
    for iter in range(config["policy_pretrain_steps"]):
        # create "random" world model = basically random transition probabilities (and random reward if learned)
        random_model.randomize()
        contextualized_policy.set_context_by_querying(random_model)
        
        # train policy on trajectories from random model
        for policy_iter in range(config["policy_inner_training_steps"]):
            trajectories = sample_trajectories(LearnedEnv(random_model, env_true.action_space, env_true.observation_space), 
                                               contextualized_policy, max_steps=config["max_steps"], num_trajectories=config["policy_trajectories_per_step"])
            trainer.train_step(trajectories, baseline)

        # TODO: how do we know we have converged? => we should do some sort of validation to see if we are still improving

    # Train model (leader)
    # NOTE: We are not using a replay buffer because then some trajectories are produced by best-response policies to older world models, violating the follower-best-reponse criteria.
    #       Even though we are only using the trajectories to learn to predict the next states given a state-action-pair, having a non-best-response state-visitation distribution, will skew the weighting in the loss, giving us a sub-optimal policy-model combination.
    # TODO: on the other hand, this means we are probably less sample efficient, so it might be worth to try both (probably it converges in the end because the world model wont change much and so the policy will be consistent and thus also the trajectories)
    for iter in range(config["training_iterations"]):
        print(f"Training iteration {iter}")

        # Sample trajectories on the environment, using the best-response-policy (wrt the current model)
        contextualized_policy.set_context_by_querying(model)
        env_trajectories = sample_trajectories(env_true, contextualized_policy, 
                                               max_steps=config["max_steps"], num_trajectories=config["init_samples"]) 

        states = np.concatenate(env_trajectories.states)
        actions = np.concatenate(env_trajectories.actions)
        rewards = np.concatenate(env_trajectories.rewards)
        next_states = np.concatenate(env_trajectories.next_states)

        average_reward = np.mean([trajectory.total_reward for trajectory in env_trajectories])
        print(f"Average reward: {average_reward}")
        
        dynamics_loss = model.fit_dynamics(states, actions, next_states, 
                                           fit_epochs=config["model_batch_size"], fit_mb_size=config["model_fit_epochs"])
        print(f"Dynamics loss: {dynamics_loss}")

        if config["learn_reward"]:
            reward_loss = model.fit_reward(states, actions, rewards, 
                                           fit_epochs=config["model_batch_size"], fit_mb_size=config["model_fit_epochs"])
            print(f"Reward loss: {reward_loss}")
        
        

if __name__ == '__main__':
    train_contextualized_MAL()
