from gymnasium import spaces
import numpy as np
import torch
from torch.nn.functional import cross_entropy, softmax
from stable_baselines3.common.policies import BasePolicy

from stackelberg_mbrl.envs.env_util import AEnv
from stackelberg_mbrl.util.tensor_util import OneHot, one_hot


class LeaderEnv(AEnv):
    """
    This is a wrapper around an environemnt to create the leader MDP.

    Each episode starts by querying the leader model (the env_model), then updating the follower (the policy with the query answers)
    and then starting the real environment.
    Observations are state, action pairs of the policy and the actions (of the model) are distributions over new states.
    The reward is the l2 divergacne of the predicted distribution and the actual next state according to the true env.
    """
    def __init__(self, true_env: AEnv, policy: BasePolicy, queries: list[tuple[OneHot, OneHot]]):
        super().__init__()
        self.true_env = true_env
        self.policy = policy
        self.queries = queries

        # An observation is a state\action pair
        self.observation_space = spaces.MultiDiscrete([true_env.observation_dim, true_env.action_dim])
        # An action is a prediction for the next state
        self.action_space = spaces.Box(-1e10, 1e10, (true_env.observation_dim, ))

    @property
    def max_episode_steps(self):
        # We have the inital segment of querying and then one episode of the true env
        return len(self.queries) + self.true_env.max_episode_steps

    @property
    def observation_dim(self):
        # Each observation is one state\action pair.
        return self.true_env.observation_dim + self.true_env.action_dim

    @property
    def action_dim(self):
        # Each action is a stachastic prediction (probability distribution) of the next observation
        return self.true_env.observation_dim
    
    def reset(self, seed: int | None = None):
        self.query_answers = []
        self.step_count = 0
        self.true_env_obs, _ = self.true_env.reset(seed=seed)
        self.next_policy_action, _ = self.policy.predict(self.true_env_obs)
        self.total_loss = 0.0

        if self.step_count < len(self.queries):
            obs = self.queries[self.step_count]
        else:
            # This branch is for the case that we have no queries at all.
            obs = (self.true_env_ob, self.next_policy_action)

        return obs, {}

    def step(self, action: torch.Tensor | np.ndarray):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        # # As action can be an arbitrary Box(3) we apply softmax to get a distribution
        next_state_prediction = softmax(action)

        # If the action was an answer to a query, save it
        if self.step_count < len(self.queries):
            self.query_answers.append(next_state_prediction)
        # If that was the last query, update the context of the policy
        if self.step_count == len(self.queries) - 1:
            self.policy.features_extractor.set_context(torch.concat(self.query_answers))

        if self.step_count < len(self.queries) - 1:
            # We are still in the querying phase
            obs = self.queries[self.step_count]
            self.step_count += 1
            reward = 0
            terminated = False
            truncated = False
            return obs, reward, terminated, truncated, {}
        else:
            # Make one step in the environment to see if the prediction of the model was accurate
            self.true_env_obs, env_reward, terminated, truncated, info = self.true_env.step(self.next_policy_action)
            # To calculate the reward, we want to know how good the prediction of the world model was.
            # Note that the `action` is the prediction of the world model.
            if self.step_count >= len(self.queries):
                # print("predicted next state", action)
                # print("actual next state", one_hot(self.true_env_obs, self.true_env.observation_dim).float())
                # print("cross entropy", cross_entropy(action, one_hot(self.true_env_obs, self.true_env.observation_dim).float()))
                
                # self.total_loss += cross_entropy(action, one_hot(self.true_env_obs, self.true_env.observation_dim).float())
                next_state_prediction = np.array(next_state_prediction)
                next_state_prediction[self.true_env_obs] -= 1
                self.total_loss += np.sum(np.square(next_state_prediction))
                # print(cross_entropy(action, one_hot(self.true_env_obs, self.true_env.observation_dim).float()))
                # print(env_reward)
                # self.total_loss -= 100*env_reward

            self.next_policy_action, _ = self.policy.predict(self.true_env_obs)

            self.step_count += 1
            reward = - self.total_loss / (self.step_count - len(self.queries) + 1) if terminated or truncated else 0
            # print("reward", reward)
            return (self.true_env_obs, self.next_policy_action), reward, terminated, truncated, info
