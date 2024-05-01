from envs.gym_env import GymEnv
from models.nn_dynamics import WorldModel
from models.model_based_npg import ModelBasedNPG
from models.gaussian_mlp import MLP
from models.mlp_baseline import MLPBaseline
from util.sampling_core import sample_data_batch
import torch
import numpy as np


def train():
    config = {
        "seed": 123,
        "policy_size": (32, 32),
        "init_log_std": -0.5,
        "min_log_std": -2.5,
        "device": "cpu",
        "npg_step_size": 0.05,
        "training_iterations": 1000,
        "init_samples": 500,
        "env_replay_buffer_size": 20000,
        "policy_inner_training_steps": 5,
        "model_batch_size": 64,
        "model_fit_epochs": 10,
        "policy_trajectories_per_step": 250,
        "max_steps": 50,
        "num_models": 4,
    }

    np.random.seed(config["seed"])
    torch.random.manual_seed(config["seed"])

    env = GymEnv("state-machine", act_repeat=1)
    env.set_seed(config["seed"])

    models = [WorldModel(state_dim=env.observation_dim, act_dim=env.action_dim, seed=config["seed"], learn_reward=True) for i in range(config['num_models'])]
    policy = MLP(env.spec, seed=config["seed"], hidden_sizes=config['policy_size'], 
                    init_log_std=config['init_log_std'], min_log_std=config['min_log_std'])

    baseline = MLPBaseline(env.spec, reg_coef=1e-3, batch_size=128, epochs=1,  learn_rate=1e-3,
                       device=config['device'])
    agent = ModelBasedNPG(learned_model=models, env=env, policy=policy, baseline=baseline, seed=config["seed"],
                      normalized_step_size=config['npg_step_size'], save_logs=True)
    

    env_replay_buffer = []
    init_states_buffer = []

    for iter in range(config["training_iterations"]):
        print(f"Training iteration {iter}")
        samples_trajectories = sample_data_batch(config["init_samples"], env, policy, eval_mode=False, base_seed=config["seed"] + iter)

        for traj in samples_trajectories:
            env_replay_buffer.append(traj)
            init_states_buffer.append(traj['observations'][0])

        if len(env_replay_buffer) > config["env_replay_buffer_size"]:
            env_replay_buffer = env_replay_buffer[-config["env_replay_buffer_size"]:]
            init_states_buffer = init_states_buffer[-config["env_replay_buffer_size"]:]
        
        states = np.concatenate([p['observations'][:-1] for p in env_replay_buffer])
        actions = np.concatenate([p['actions'][:-1] for p in env_replay_buffer])
        next_states = np.concatenate([p['observations'][1:] for p in env_replay_buffer])
        rewards = np.concatenate([p['rewards'][:-1] for p in env_replay_buffer])

        average_reward = np.mean([np.sum(p['rewards']) for p in samples_trajectories])

        # Train models
        for model in models:
            model_loss = model.fit_dynamics(states, actions, next_states, fit_epochs=config["model_batch_size"], fit_mb_size=config["model_fit_epochs"], set_transformations=False)

        # Train policy
        for policy_iter in range(config["policy_inner_training_steps"]):
            
            buffer_rand_idx = np.random.choice(len(init_states_buffer), size=config['policy_trajectories_per_step'], replace=True).tolist()
            init_states = [init_states_buffer[idx] for idx in buffer_rand_idx]

            agent.train_step(config["policy_trajectories_per_step"], init_states=init_states, horizon=config["max_steps"])

        print(f"Average reward: {average_reward}")
        print(f"Model loss: {model_loss}")
        

if __name__ == '__main__':
    train()
