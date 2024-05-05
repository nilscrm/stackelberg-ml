from typing import List
import numpy as np
import gym
from nn.baseline.baselines import ABaseline
from policies.policy import APolicy


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
        # TODO: should we do this for each trajectory separately and return a list? because not all of them have the same length...
        observations = np.concatenate(self.states)
        expected_returns = baseline.predict_expected_returns(observations)

        if gae_lambda is None or gae_lambda < 0.0 or gae_lambda > 1.0:
            # Standard Advantage Computation
            returns = np.array([trajectory.compute_discounted_rewards(gamma) for trajectory in self.trajectories])
            advantages = returns - expected_returns
        else:
            # Generalized Advantage Estimation (GAE)
            # append 0 if terminated, repeat last reward if not terminated
            expected_returns = np.concatenate([expected_returns, np.zeros(self.num_trajectories)])
            expected_returns[:,-1] += (np.concatenate(self.terminated) == False) * expected_returns[:,-2]

            td_deltas = np.concatenate(self.rewards) + gamma*expected_returns[:,1:] - expected_returns[:,:-1]            
            advantages = compute_discounted_rewards(td_deltas, gamma*gae_lambda)

        if normalize:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            return advantages

    


########################
###     Sampling     ###
########################

def sample_trajectories(env: gym.Env, policy: APolicy, num_trajectories: int, max_steps: int | None = None) -> TrajectoryList:
    return TrajectoryList([sample_trajectory(env, policy, max_steps) for i in range(num_trajectories)])

def sample_trajectory(env: gym.Env, policy: APolicy, max_steps: int | None = None):
    """ Sample a trajectory in an environment using a policy """

    state = env.reset()
    done = False
    steps = 0

    states = []
    actions = []
    rewards = []
    next_states = []

    while not done and (max_steps is None or steps <= max_steps):
        print(type(env))
        print(state)
        action, _ = policy.sample_next_action(state)
        print(action)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)

        state = next_state
        steps += 1

    return Trajectory(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), done)



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