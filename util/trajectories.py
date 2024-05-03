from typing import List
import numpy as np
import gym
from policies.policy import APolicy


class Trajectory:
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
    


########################
###     Sampling     ###
########################

def sample_trajectories(env: gym.Env, policy: APolicy, num_trajectories: int, max_steps: int | None = None):
    return [sample_trajectory(env, policy, max_steps) for i in range(num_trajectories)]

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
        action, _ = policy.get_action(state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)

        state = next_state
        steps += 1

    return Trajectory(np.array(states), np.array(actions), np.array(rewards), np.array(next_states))



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
    

def compute_advantages(trajectories: List[Trajectory], baseline_model, gamma, gae_lambda=None, normalize=False):
    # TODO: figure out what the f**k baseline does? is it just predicting the expected cumulative reward starting in some state?
    baselines = np.array([baseline_model.predict(trajectory) for trajectory in trajectories])

    if gae_lambda is None or gae_lambda < 0.0 or gae_lambda > 1.0:
        # Standard Advantage Computation
        returns = np.array([trajectory.compute_discounted_rewards(gamma) for trajectory in trajectories])
        advantages = returns - baselines
    else:
        # Generalized Advantage Estimation (GAE)
        # TODO: when is b.ndim == 1?
        for trajectory in trajectories:
            b = trajectory["baseline"] = baseline.predict(trajectory)
            if b.ndim == 1:
                b1 = np.append(trajectory["baseline"], 0.0 if trajectory["terminated"] else b[-1])
            else:
                b1 = np.vstack((b, np.zeros(b.shape[1]) if trajectory["terminated"] else b[-1]))
            td_deltas = trajectory["rewards"] + gamma*b1[1:] - b1[:-1]
        
        advantages = compute_discounted_rewards(td_deltas, gamma*gae_lambda)

    if normalize:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    else:
        return advantages