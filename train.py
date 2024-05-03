from envs.gym_env import GymEnv
from envs.learned_env import LearnedEnv
from models.nn_dynamics import WorldModel
from models.random_dynamics import RandomWorldModel
from models.model_based_npg import ModelBasedNPG
from models.gaussian_mlp import MLP
from models.mlp_baseline import MLPBaseline
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
    model = GymEnv(LearnedEnv(WorldModel(state_dim=env_true.observation_dim, act_dim=env_true.action_dim, seed=config["seed"], learn_reward=config["learn_reward"])))
    random_model = RandomWorldModel(state_dim=env_true.observation_dim, act_dim=env_true.action_dim, seed=config["seed"], learn_reward=config["learn_reward"])

    # context = in which state will we land (+ what reward we get) for each query
    queries = product(range(env_true.observation_dim), range(env_true.action_dim))
    context_size = len(queries) * (env_true.observation_dim + 1 if config["learn_reward"] else 0)
    policy = MLP(env_true.spec, seed=config["seed"], hidden_sizes=config['policy_size'], 
                    init_log_std=config['init_log_std'], min_log_std=config['min_log_std'],
                    context_size=context_size)
    contextualized_policy = ModelContextualizedPolicy(policy, queries)

    baseline = MLPBaseline(env_true.spec, reg_coef=1e-3, batch_size=128, epochs=1,  learn_rate=1e-3, device=config['device'])
    # TODO: rewrite ModelBasedNPG such that it does not require env_true, bc it should not!
    trainer = ModelBasedNPG(env=env_true, policy=contextualized_policy, baseline=baseline, normalized_step_size=config['npg_step_size'], save_logs=True)
    
    # Pretrain the policy conditioned on a world model
    for iter in range(config["policy_pretrain_steps"]):
        # create "random" world model = basically random transition probabilities (and random reward if learned)
        random_model.randomize()
        contextualized_policy.set_context_by_querying(random_model)
        
        # train policy on trajectories from random model
        for policy_iter in range(config["policy_inner_training_steps"]):
            trajectories = sample_trajectories(LearnedEnv(random_model), contextualized_policy, 
                                               max_steps=config["max_steps"], num_trajectories=config["policy_trajectories_per_step"])
            trainer.train_step(trajectories)

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

        states = np.concatenate([trajectory.states for trajectory in env_trajectories])
        actions = np.concatenate([trajectory.actions for trajectory in env_trajectories])
        rewards = np.concatenate([trajectory.rewards for trajectory in env_trajectories])
        next_states = np.concatenate([trajectory.next_states for trajectory in env_trajectories])

        average_reward = np.mean([trajectory.total_reward for trajectory in env_trajectories])
        print(f"Average reward: {average_reward}")
        
        dynamics_loss = model.fit_dynamics(states, actions, next_states, fit_epochs=config["model_batch_size"], fit_mb_size=config["model_fit_epochs"], set_transformations=False)
        print(f"Dynamics loss: {dynamics_loss}")

        if config["learn_reward"]:
            reward_loss = model.fit_reward(states, actions, rewards, fit_epochs=config["model_batch_size"], fit_mb_size=config["model_fit_epochs"], set_transformations=False)
            print(f"Reward loss: {reward_loss}")
        
        

if __name__ == '__main__':
    train_contextualized_MAL()
