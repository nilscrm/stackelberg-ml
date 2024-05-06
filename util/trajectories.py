from typing import List
import numpy as np
import gymnasium
import torch
from envs.env_util import AEnv
from nn.baseline.baselines import ABaseline
from policies.policy import APolicy
from util.tensor_util import one_hot, one_hot_to_idx


class Trajectory:
    # TODO: should we also store the context we used?
    def __init__(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray, terminated: bool):
        self.length = states.shape[0]
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards
        self.terminated = terminated

        self.total_reward = np.sum(self.rewards)

    def compute_discounted_rewards(self, gamma = 1.0, terminal = 0.0):
        return compute_discounted_rewards(self.rewards, gamma, terminal)
    
    def compute_advantage(self, baseline: ABaseline, gamma, gae_lambda=None):
        expected_returns = baseline.predict_expected_returns(self.states).detach().numpy()

        if gae_lambda is None or gae_lambda < 0.0 or gae_lambda > 1.0:
            # Standard Advantage Computation
            returns = self.compute_discounted_rewards(gamma)
            advantages = returns - expected_returns
        else:
            # Generalized Advantage Estimation (GAE)
            # append 0 if terminated, repeat last reward if not terminated
            expected_returns = np.append(expected_returns, 0 if self.terminated else expected_returns[-1])

            td_deltas = self.rewards + gamma*expected_returns[1:] - expected_returns[:-1]
            advantages = compute_discounted_rewards(td_deltas, gamma*gae_lambda)
        
        return advantages
    
    def to_string(self, state_names, action_names):
        state_idx = np.argmax(self.states, axis=-1)
        action_idx = np.argmax(self.actions, axis=-1)
        next_state_idx = np.argmax(self.next_states, axis=-1)
        as_str = state_names[state_idx[0]]
        for (s,a,s_next,r) in zip(state_idx, action_idx, next_state_idx, self.rewards):
            as_str += f"-{action_names[a]}-({r})->{state_names[s_next]}"
        return as_str


class TrajectoryList:
    def __init__(self, trajectories: List[Trajectory]):
        self.trajectories = trajectories
    
    @property
    def states(self):
        return [trajectory.states for trajectory in self.trajectories]
    
    @property
    def actions(self):
        return [trajectory.actions for trajectory in self.trajectories]
    
    @property
    def rewards(self):
        return [trajectory.rewards for trajectory in self.trajectories]
    
    @property
    def terminated(self):
        return [trajectory.terminated for trajectory in self.trajectories]

    @property
    def total_rewards(self):
        return [trajectory.total_reward for trajectory in self.trajectories]
    
    @property
    def next_states(self):
        return [trajectory.next_states for trajectory in self.trajectories]
    
    @property
    def num_trajectories(self):
        return len(self.trajectories)
    
    @property
    def num_samples(self):
        return np.sum([trajectory.length for trajectory in self.trajectories])
    
    def __getitem__(self, index) -> Trajectory:
        return self.trajectories[index]

    def compute_discounted_rewards(self, gamma = 1.0, terminal = 0.0):
        return [trajectory.compute_discounted_rewards(gamma, terminal) for trajectory in self.trajectories]

    def compute_advantages(self, baseline: ABaseline, gamma, gae_lambda=None, normalize=False):
        advantages = [trajectory.compute_advantage(baseline, gamma, gae_lambda) for trajectory in self.trajectories]

        if normalize:
            all_advantages = np.concatenate(advantages)
            advantage_mean = all_advantages.mean()
            advantage_std = all_advantages.std() + 1e-8

            for i in range(len(advantages)):
                advantages[i] = (advantages[i] - advantage_mean) / (advantage_std)

        return advantages

    


########################
###     Sampling     ###
########################

def sample_trajectories(env: AEnv, policy: APolicy, num_trajectories: int, max_steps: int | None = None) -> TrajectoryList:
    return TrajectoryList([sample_trajectory(env, policy, max_steps) for i in range(num_trajectories)])

def sample_trajectory(env: AEnv, policy: APolicy, max_steps: int | None = None):
    """ Sample a trajectory in an environment using a policy """

    state, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0

    states = []
    actions = []
    next_states = []
    rewards = []

    with torch.no_grad():
        while (not terminated) and (not truncated) and (max_steps is None or steps <= max_steps):
            action = one_hot_to_idx(policy.sample_next_action(one_hot(state, env.observation_dim)))
            next_state, reward, terminated, truncated, info = env.step(action)

            states.append(one_hot(state, env.observation_dim))
            actions.append(one_hot(action, env.action_dim))
            next_states.append(one_hot(next_state, env.observation_dim))
            rewards.append(reward)

            state = next_state
            steps += 1

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        next_states = np.stack(next_states, axis=0)
        rewards = np.array(rewards)

    return Trajectory(states, actions, next_states, rewards, terminated)



########################
###       Util       ###
########################

def compute_discounted_rewards(rewards: np.ndarray, gamma = 1.0, terminal = 0.0) -> np.ndarray:
    """ 
        Compute the discounted rewards in a numerically stable way. 
        The discounted rewards are an array, where element i contains the reward if the trajectory would have started from there.
    """
    assert rewards.ndim == 1

    discounted_rewards = np.zeros_like(rewards)

    run_sum = terminal
    for t in reversed(range(len(rewards))):
        run_sum = rewards[t] + gamma*run_sum
        discounted_rewards[t] = run_sum

    return discounted_rewards