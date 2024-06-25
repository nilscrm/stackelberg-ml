from typing import List, Callable, Any
import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from stackelberg_mbrl.nn.baseline.baselines import ABaseline
from stackelberg_mbrl.policies.policy import APolicy
from stackelberg_mbrl.util.tensor_util import one_hot, one_hot_to_idx
from stable_baselines3.common.policies import ActorCriticPolicy


class Trajectory:
    # TODO: should we also store the context we used?
    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminated: bool, num_actions: int, num_states: int):
        self.length = states.shape[0] - 1
        self._states = states
        self.actions = actions
        self.rewards = rewards
        assert self.length == self.actions.shape[0]
        assert self.length == self.rewards.shape[0]

        self.terminated = terminated
        self.num_actions = num_actions
        self.num_states = num_states
        self.total_reward = np.sum(self.rewards)

    @property
    def states(self):
        return self._states[:-1]
    
    @property
    def next_states(self):
        return self._states[1:]
    
    @property
    def query_shape(self):
        return (self.num_states, self.num_actions, self.num_states)

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
    
    def to_query_tensor(self):
        next_states_OH = torch.tensor(self.next_states).long()
        next_states_OH = F.one_hot(next_states_OH, num_classes=self.num_states).float()

        query_tensor = torch.zeros(self.num_states, self.num_actions, self.num_states)
        mask = torch.zeros(self.num_states, self.num_actions)

        for s, a, oh in zip(self.states, self.actions, next_states_OH):
            query_tensor[s, a] += oh
            mask[s, a] += 1

        # TODO also try softmax
        return query_tensor, mask

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
    def total_length(self):
        return np.sum([t.length for t in self.trajectories])
    
    @property
    def next_states(self):
        return [trajectory.next_states for trajectory in self.trajectories]
    
    @property
    def num_trajectories(self):
        return len(self.trajectories)
    
    @property
    def query_shape(self):
        return self.trajectories[0].query_shape
    
    def to_query_tensor(self):
        big = [trajectory.to_query_tensor() for trajectory in self.trajectories]
        query_tensors = [x[0] for x in big]
        masks = [x[1] for x in big]
        return torch.sum(torch.stack(query_tensors), dim=0), torch.sum(torch.stack(masks), dim=0)
        
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

def sample_trajectories(env: gymnasium.Env, policy: APolicy | ActorCriticPolicy, num_trajectories: int, max_steps: int | None = None, context: np.ndarray | None = None, w_one_hot: bool = True) -> TrajectoryList:
    return TrajectoryList([sample_trajectory(env=env, policy=policy, context=context, max_steps=max_steps, w_one_hot=w_one_hot) for i in range(num_trajectories)])

def sample_trajectory(env: gymnasium.Env, policy: APolicy | ActorCriticPolicy, max_steps: int | None = None, context: np.ndarray | None = None, w_one_hot: bool = True):
    """ Sample a trajectory in an environment using a policy """

    state, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0

    states = [state]
    actions = []
    # next_states = []
    rewards = []

    if hasattr(policy, 'num_actions'):
        num_actions = policy.num_actions
    elif hasattr(env, 'num_actions'):
        num_actions = env.num_actions
    else:
        raise ValueError("cannot find num_actions anywhere")

    with torch.no_grad():
        while (not terminated) and (not truncated) and (max_steps is None or steps <= max_steps):
            match policy:
                case APolicy():
                    action_idx = policy.sample_next_action(state)
                case ActorCriticPolicy():
                    action, _ = policy.predict(np.concatenate((context, one_hot(state, env.num_states))))
                    action_idx = np.argmax(action)
                case _:
                    raise ValueError("Policy must be a callable or a policy object")
            next_state, reward, terminated, truncated, info = env.step(action_idx)

            if w_one_hot:
                actions.append(one_hot(action_idx, num_actions))
            else:
                actions.append(action_idx)
            states.append(next_state)
            rewards.append(reward)

            state = next_state
            steps += 1

        states = np.stack(states, axis=0)
        actions = np.stack(actions, axis=0)
        rewards = np.array(rewards)

    return Trajectory(states, actions, rewards, terminated, num_actions, env.num_states)


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