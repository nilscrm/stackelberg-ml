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

        if self.step_count < len(self.queries) - 1:
            # The action is an answer to a query and we have more queries to do
            self.query_answers.append(next_state_prediction)
            query = self.queries[self.step_count]
            self.step_count += 1
            return query, 0, False, False, {}
        elif self.step_count == len(self.queries) - 1:
            # The action in the answer to the last query
            # This is the beginning of the real environment
            self.query_answers.append(next_state_prediction)
            self.policy.features_extractor.set_context(torch.concat(self.query_answers))
            
            true_env_obs, _ = self.true_env.reset()
            self.policy_action, _ = self.policy.predict(true_env_obs)

            self.step_count += 1
            return (true_env_obs, self.policy_action), 0, False, False, {}
        else:
            # Step real environment with the policy action decided before
            true_env_obs, _, terminated, truncated, info = self.true_env.step(self.policy_action)
            # Calculate reward based on l2 divergence between the next state and the predicted next state
            next_state_prediction = np.array(next_state_prediction)
            next_state_prediction[true_env_obs] -= 1
            self.total_loss += np.sum(np.square(next_state_prediction))

            # Determine next policy action
            self.policy_action, _ = self.policy.predict(true_env_obs)

            # If this is the last step give the model reward based on the average prediction loss
            # We do an average instead of a sum so that the model is not encouraged to play short games.
            reward = - self.total_loss / (self.step_count - len(self.queries) + 1) if terminated or truncated else 0
            self.step_count += 1
            return (true_env_obs, self.policy_action), reward, terminated, truncated, info
