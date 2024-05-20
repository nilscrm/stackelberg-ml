import gymnasium
from gymnasium import spaces
import numpy as np
import torch
from torch.functional import F
from typing import Any
from stable_baselines3.common.policies import BasePolicy

from stackelberg_mbrl.envs.env_util import WorldModel
from stackelberg_mbrl.utils import one_hot

# We only handle discrete envs for now
State = int
Action = int


class ModelQueryingEnv(gymnasium.Env):
    """This is a wrapper for world model in which a policy can play. It additionally gets the model as queried context."""

    def __init__(self, world_model: WorldModel, queries: list[tuple[State, Action]]):
        """Wraps a world model and asks it a list of state/action queries before playing in it."""
        self.world_model = world_model
        self.queries = queries

        self.num_states = world_model.observation_space.n

        # One observation is the context which is all answers to the queries plus one one-hot-encoded current state
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=((len(queries) + 1) * self.num_states,))
        self.action_space = world_model.action_space

    def _get_obs(self, model_state: State):
        return np.concatenate((self.query_answers, one_hot(model_state, self.num_states)))

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        model_state, reward, terminated, truncated, info = self.world_model.step(action)
        return self._get_obs(model_state), reward, terminated, truncated, info

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        model_state, info = self.world_model.reset(seed=seed)
        # print("transitions matrix:", self.world_model.transition_probabilities)
        self.query_answers = np.concatenate([self.world_model.next_state_distribution(state, action) for (state, action) in self.queries])
        # print("query answers:", self.query_answers)
        return self._get_obs(model_state), info

    def render(self):
        return self.world_model.render()

    def close(self):
        self.world_model.close()


class LeaderEnv(gymnasium.Env):
    """
    This is a wrapper around an environemnt to create the leader MDP.

    Each episode starts by querying the leader model (the env_model), then updating the follower (the policy with the query answers)
    and then starting the real environment.
    Observations are state, action pairs of the policy and the actions (of the model) are distributions over new states.

    - Reward: the reward is given at the end of a trajectory and is equal to...
        (l2) ... the average MSE (between the predicted distribution and the actual next state according to the true env) over all the steps of the trajectory
        (kldiv) ... the expectation over the state-action-visitation-frequency of the kldiv between the actual transition probabilities and the true transition probabilities (approximated by the samples we collected)
    """

    def __init__(self, true_env: gymnasium.Env[int, int], policy: BasePolicy, queries: list[tuple[int, int]], reward: str = "kldiv"):
        self.true_env = true_env
        self.policy = policy
        self.queries = queries
        self.reward = reward

        self.true_env_num_states = true_env.observation_space.n
        self.true_env_num_actions = true_env.action_space.n

        # An observation is a state\action pair
        self.observation_space = spaces.MultiDiscrete([self.true_env_num_states, self.true_env_num_actions])
        # An action is a prediction for the next state
        self.action_space = spaces.Box(-1e10, 1e10, (self.true_env_num_states,))

        self.true_P_traj_total = np.zeros((self.true_env_num_states, self.true_env_num_actions, self.true_env_num_states))
        self.model_P_traj_total = np.zeros((self.true_env_num_states, self.true_env_num_actions, self.true_env_num_states))
        self.obs_P_cnt = np.zeros((self.true_env_num_states, self.true_env_num_actions))

    def reset(self, seed: int | None = None):
        self.query_answers = []
        self.step_count = 0
        self.total_loss = 0.0

        self.true_P_traj_total[...] = 0.0
        self.model_P_traj_total[...] = 0.0
        self.obs_P_cnt[...] = 0.0

        if self.step_count < len(self.queries):
            obs = self.queries[self.step_count]
        else:
            raise NotImplemented("LeaderEnv can't handle the case of no queries yet.")

        return obs, {}

    def step(self, action: np.ndarray):
        # An action is a prediction over the next state
        action = torch.from_numpy(action)
        next_state_prediction = F.softmax(action, dim=-1)

        if self.step_count < len(self.queries) - 1:
            # The action is an answer to a query and we have more queries to do
            self.query_answers.append(next_state_prediction)
            self.step_count += 1
            query = self.queries[self.step_count]
            return query, 0, False, False, {}
        elif self.step_count == len(self.queries) - 1:
            # The action in the answer to the last query
            # This is the beginning of the real environment
            self.query_answers.append(next_state_prediction)
            self.query_answers = np.concatenate(self.query_answers)

            true_env_obs, _ = self.true_env.reset()
            self.policy_action, _ = self.policy.predict(
                np.concatenate((self.query_answers, one_hot(true_env_obs, self.true_env_num_states)))
            )

            self.current_state = true_env_obs
            self.step_count += 1
            return (true_env_obs, self.policy_action), 0, False, False, {}
        else:
            # Step real environment with the policy action decided before
            true_env_obs, _, terminated, truncated, info = self.true_env.step(self.policy_action)

            reward = 0.0
            if self.reward == "l2":
                # Calculate reward based on l2 divergence between the next state and the predicted next state
                next_state_prediction = np.array(next_state_prediction)
                next_state_prediction[true_env_obs] -= 1
                self.total_loss += np.sum(np.square(next_state_prediction))

                # If this is the last step give the model reward based on the average prediction loss
                # We do an average instead of a sum so that the model is not encouraged to play short games.
                if terminated or truncated:
                    reward = -self.total_loss / (self.step_count - len(self.queries) + 1)
            elif self.reward == "kldiv":            
                s = self.current_state
                a = self.policy_action

                self.true_P_traj_total[s, a] += one_hot(true_env_obs, self.true_env_num_states)
                self.model_P_traj_total[s, a] += next_state_prediction.numpy()
                self.obs_P_cnt[s, a] += 1.0

                if terminated or truncated:
                    with torch.no_grad():
                        true_total = torch.tensor(self.true_P_traj_total)
                        model_total = torch.tensor(self.model_P_traj_total)
                        obs_cnt = torch.tensor(self.obs_P_cnt)

                        # convert to estimates P(-|s,a) forall s,a
                        true_P = true_total / obs_cnt.unsqueeze(-1)
                        model_P = model_total / obs_cnt.unsqueeze(-1)

                        # compute D_KL(P(-|s,a), \hat{P}(-|s,a)) forall s,a
                        kl_divs = (true_P * (true_P / (model_P + 1e-5)).log()).sum(-1)

                        # expectation according to state-action-visitation-frequency (s,a without observations will be 0 => inf/nan)
                        reward = -torch.nan_to_num(((obs_cnt / obs_cnt.sum()) * kl_divs), nan=0.0, posinf=0.0, neginf=0.0).sum().item()

            # Determine next policy action
            self.policy_action, _ = self.policy.predict(
                np.concatenate((self.query_answers, one_hot(true_env_obs, self.true_env_num_states)))
            )

            self.current_state = true_env_obs
            self.step_count += 1
            return (true_env_obs, self.policy_action), reward, terminated, truncated, info
