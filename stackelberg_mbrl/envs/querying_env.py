import gymnasium
from gymnasium import spaces
import numpy as np
from stackelberg_mbrl.nn.model.world_models import ContextualizedWorldModel
from stackelberg_mbrl.policies.policy import APolicy
import torch
from torch.functional import F
from typing import Any
from stable_baselines3.common.policies import BasePolicy

from stackelberg_mbrl.envs.env_util import WorldModel
from stackelberg_mbrl.utils import one_hot

# We only handle discrete envs for now
State = int
Action = int

class CountedEnvWrapper(gymnasium.Env):
    """This is a wrapper which counts how much data is collected from the environment (trajectories and samples)"""
    def __init__(self, env: gymnasium.Env):
        self.env = env
        self.trajectories = 0
        self.samples = 0

    def __getattribute__(self, name: str) -> Any:
        # This is a magic method that is called when you use the dot operator on an object.
        # We use it to delegate all calls (excluding the ones in the following dict) to the wrapped environment
        if name in {"step", "reset", "env", "trajectories", "samples"}:
            return super().__getattribute__(name)
        else:
            return getattr(self.env, name)

    def step(self, action: Any):
        ret = self.env.step(action)
        self.samples += 1
        return ret
    
    def reset(self, seed: int | None = None):
        ret = self.env.reset(seed=seed)
        self.trajectories += 1
        self.samples += 1
        return ret
    

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


class ConstantContextEnv(gymnasium.Env):
    """This is a wrapper for world model in which a policy can play. It additionally gets the model as queried context."""

    def __init__(self, env: gymnasium.Env, context: np.ndarray):
        """Wraps a world model and asks it a list of state/action queries before playing in it."""
        self.env = env
        self.context = context

        self.num_states = env.observation_space.n

        # One observation is the context which is all answers to the queries plus one one-hot-encoded current state
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(context) + self.num_states,))
        self.action_space = env.action_space

    def _get_obs(self, model_state: State):
        return np.concatenate((self.context, one_hot(model_state, self.num_states)))

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        model_state, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(model_state), reward, terminated, truncated, info

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        model_state, info = self.env.reset(seed=seed)
        return self._get_obs(model_state), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class PolicyQueryingEnv(gymnasium.Env):
    # TODO: merge with ModelQueryingEnv
    """This is a wrapper for an environment in which a policy can play. It additionally gets the policy as queried context."""

    def __init__(self, env: gymnasium.Env, queries: list[State], before_reset: callable = None, policy: APolicy = None):
        """Wraps an environment and asks the policy a list of state queries before playing in it. Allows for custom callable-action before reset. """
        self.env = env
        self.policy = policy
        self.queries = queries
        self.before_reset = before_reset

        self.num_states = env.observation_space.n

        # One observation is the context which is all answers to the queries plus one one-hot-encoded current state
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=((len(queries) + 1) * self.num_states,))
        self.action_space = env.action_space

    def set_policy(self, policy: APolicy):
        self.policy = policy
        self.update_query_answers()

    def update_query_answers(self):
        self.query_answers = np.concatenate([self.policy.next_action_distribution(state) for state in self.queries])

    def _get_obs(self, model_state: State):
        return np.concatenate((self.query_answers, one_hot(model_state, self.num_states)))

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        model_state, reward, terminated, truncated, info = self.env.step(action)
        return self._get_obs(model_state), reward, terminated, truncated, info

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if self.before_reset:
            self.before_reset(seed)
        
        self.update_query_answers()
        model_state, info = self.env.reset(seed=seed)
        return self._get_obs(model_state), info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class LeaderEnv(gymnasium.Env):
    """
    This is a wrapper around an environemnt to create the leader MDP.

    Each episode starts by querying the leader model (the env_model), then updating the follower (the policy with the query answers)
    and then starting the real environment.
    Observations are state, action pairs of the policy and the actions (of the model) are distributions over new states.

    env_reward_weight: How much weight will be given to the reward signal from the true environment vs the error the model makes on the predictions.
        (e.g. 0 corresponds to just standard l2 divergence of the predicted distribution and the actual next state according to the true env,
              1 corresponds to just the reward that the policy achieves in the true env)
    """

    def __init__(self, true_env: gymnasium.Env[int, int], policy: BasePolicy, queries: list[tuple[int, int]], env_reward_weight: float = 0.0):
        self.true_env = true_env
        self.policy = policy
        self.queries = queries
        self.env_reward_weight = env_reward_weight

        self.true_env_num_states = true_env.observation_space.n
        self.true_env_num_actions = true_env.action_space.n

        # An observation is a state\action pair
        self.observation_space = spaces.MultiDiscrete([self.true_env_num_states, self.true_env_num_actions])
        # An action is a prediction for the next state
        self.action_space = spaces.Box(-1e10, 1e10, (self.true_env_num_states,))

    def reset(self, seed: int | None = None):
        self.query_answers = []
        self.step_count = 0
        self.total_loss = 0.0

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

            self.step_count += 1
            return (true_env_obs, self.policy_action), 0, False, False, {}
        else:
            # Step real environment with the policy action decided before
            true_env_obs, env_reward, terminated, truncated, info = self.true_env.step(self.policy_action)
            # Calculate reward based on l2 divergence between the next state and the predicted next state
            next_state_prediction = np.array(next_state_prediction)
            next_state_prediction[true_env_obs] -= 1
            self.total_loss += np.sum(np.square(next_state_prediction))

            # Determine next policy action
            self.policy_action, _ = self.policy.predict(
                np.concatenate((self.query_answers, one_hot(true_env_obs, self.true_env_num_states)))
            )

            reward = self.env_reward_weight * env_reward
            if terminated or truncated:
                # If this is the last step give the model reward based on the average prediction loss
                # We do an average instead of a sum so that the model is not encouraged to play short games.
                reward -= (1.0 - self.env_reward_weight) * self.total_loss / (self.step_count - len(self.queries) + 1)

            self.step_count += 1

            return (true_env_obs, self.policy_action), reward, terminated, truncated, info


class PALLeaderEnv(gymnasium.Env[int,int]):
    """
        This is a wrapper around a contextualized world model that the leader can play in.
        First, all the queries will be returned with 0 reward, before starting to simulate an environment using a learned contextualized model
    """
    def __init__(self, ctx_world_model: ContextualizedWorldModel, initial_state: int, queries: list[int], max_ep_steps: int, final_state: int = None):
        self.ctx_world_model = ctx_world_model
        self.initial_state = initial_state
        self.final_state = final_state
        self.max_ep_steps = max_ep_steps
        self.queries = queries

        self.observation_space = spaces.Discrete(ctx_world_model.observation_dim)
        self.action_space = spaces.Discrete(ctx_world_model.action_dim)

    def set_policy(self, policy: BasePolicy):
        self.policy = policy
        self.update_context()

    def update_context(self):
        self.context = np.concatenate([one_hot(self.policy.predict(query)[0], self.ctx_world_model.action_dim) for query in self.queries])
        self.context_size = self.context.shape[0]

    def reset(self, seed: int | None = None):
        self.update_context() # TODO: dont update on every reset but instead have callback in PPO after training step
        self.step_count = 0
        return self.queries[0], {}

    def step(self, action_idx: int):
        if self.step_count < len(self.queries):
            # query
            observation = self.queries[self.step_count] # TODO: one_hot?
            self.step_count += 1
            return observation, 0, False, False, {}
        elif self.step_count == len(self.queries):
            # reset on env
            self.current_state_idx = self.initial_state
            self.step_count += 1
            return self.initial_state, 0, False, False, {}
        else:
            # step on env
            observation = np.concatenate([self.context, one_hot(self.current_state_idx, self.ctx_world_model.observation_dim)])
            action = one_hot(action_idx, self.ctx_world_model.action_dim)
            observation_next = self.ctx_world_model.sample_next_state(observation, action)
            state_next = observation_next[self.context_size:]
            state_next_idx = state_next.argmax().item()

            reward = self.ctx_world_model.reward(observation, action, observation_next)

            self.step_count += 1
            self.current_state_idx = state_next_idx

            terminated = (state_next_idx == self.final_state)
            truncated = (self.step_count >= self.max_ep_steps)
        
            return state_next_idx, reward, terminated, truncated, {}
