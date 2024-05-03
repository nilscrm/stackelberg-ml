from envs.gym_env import GymEnv
from models.nn_dynamics import WorldModel
from models.random_dynamics import RandomWorldModel
from models.model_based_npg import ModelBasedNPG
from models.gaussian_mlp import MLP
from models.mlp_baseline import MLPBaseline
import torch
import numpy as np

from util.sampling import sample_model_trajectory, sample_env_trajectory

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

    env = GymEnv("state-machine", act_repeat=1)
    env.set_seed(config["seed"])

    queries = product(range(env.observation_dim), range(env.action_dim))

    # NOTE: in this scenario it does not make sense to have multiple world models, as they would all converge to a stackelberg equilibrium and not help to find the best policy
    model = WorldModel(state_dim=env.observation_dim, act_dim=env.action_dim, seed=config["seed"], learn_reward=config["learn_reward"])
    random_model = RandomWorldModel(state_dim=env.observation_dim, act_dim=env.action_dim, seed=config["seed"], learn_reward=config["learn_reward"])

    policy = MLP(env.spec, seed=config["seed"], hidden_sizes=config['policy_size'], 
                    init_log_std=config['init_log_std'], min_log_std=config['min_log_std'],
                    num_queries=len(queries)*config['num_models']) # TODO: should also somehow include reward

    baseline = MLPBaseline(env.spec, reg_coef=1e-3, batch_size=128, epochs=1,  learn_rate=1e-3,
                       device=config['device'])
    agent = ModelBasedNPG(learned_model=[model], env=env, policy=policy, baseline=baseline, seed=config["seed"],
                      normalized_step_size=config['npg_step_size'], save_logs=True)
    

    # Pretrain the policy conditioned on a world model
    for iter in range(config["policy_pretrain_steps"]):
        trajectories = []
        # create "random" world model = basically random transition probabilities (and random reward if learned)
        random_model.randomize()
        
        # sample trajectories from random model
        for _ in range(config["policy_trajectories_per_step"]):
            init_state = env.reset()
            reward_function = None
            termination_function = None

            trajectory = sample_model_trajectory(random_model, policy, queries, init_state, reward_function, termination_function, config["max_steps"])
            
            trajectories.append(trajectory)
        
        # train policy on trajectories from random model
        model_descriptor = None # contextualization of the world model that the policy is conditioned on

        for policy_iter in range(config["policy_inner_training_steps"]):
            init_states = [trajectory[0] for trajectory in trajectories] # only use init_states from trajectories that are conditioned on the current random world model

            # TODO: are we using trajectories only for init_states here??? this seems wrong...
            # TODO: make sure the training_step actually uses trajectories from the random model
            agent.train_step(config["policy_trajectories_per_step"], init_states=init_states, horizon=config["max_steps"])

        # TODO: how do we know we have converged?


    # Train model (leader)
    # NOTE: We are not using a replay buffer because then some trajectories are produced by best-response policies to older world models, violating the follower-best-reponse criteria.
    #       Even though we are only using the trajectories to learn to predict the next states given a state-action-pair, having a non-best-response state-visitation distribution, will skew the weighting in the loss, giving us a sub-optimal policy-model combination.
    # TODO: on the other hand, this means we are probably less sample efficient, so it might be worth to try both (probably it converges in the end because the world model wont change much and so the policy will be consistent and thus also the trajectories)
    for iter in range(config["training_iterations"]):
        print(f"Training iteration {iter}")

        # Sample trajectories on the environment, using the best-response-policy (wrt the current model)
        # Format: trajectory_idx x step_idx x (state, action, reward, next_state)
        env_trajectories = np.array([
            sample_env_trajectory(env, model, policy, queries, max_steps=config["max_steps"]) 
            for i in range(config["init_samples"])
        ])

        average_reward = np.mean(np.sum(env_trajectories[:,:,2], axis=1), axis=0)
        
        flat_trajectories = np.array(env_trajectories).reshape(-1, 4)

        states = flat_trajectories[:,0]
        actions = flat_trajectories[:,1]
        next_states = flat_trajectories[:,3]

        model_loss = model.fit_dynamics(states, actions, next_states, fit_epochs=config["model_batch_size"], fit_mb_size=config["model_fit_epochs"], set_transformations=False)

        print(f"Average reward: {average_reward}")
        print(f"Model loss: {model_loss}")
        

if __name__ == '__main__':
    train_contextualized_MAL()
