from abc import ABC, abstractmethod
from collections import defaultdict
import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
import numpy as np
from pathlib import Path
import pygraphviz as pgv
from stable_baselines3.common.policies import BasePolicy
import torch
from torch.functional import F
from typing import Literal


class WorldModel(ABC, gymnasium.Env):
    """A `WorldModel` is a playable environment with the additional capapility of giving stochastic prediction over the next state."""

    observation_space: spaces.Discrete
    action_space: spaces.Discrete

    @abstractmethod
    def next_state_distribution(self, observation: int, action: int) -> np.ndarray:
        """Returns a probability distribution over next states."""
        ...


class MatrixMDP(WorldModel):
    def __init__(
        self, max_ep_steps: int, transition_probabilities: np.ndarray, rewards: np.ndarray, initial_state: int, final_state: int = -1
    ):
        self.max_ep_steps = max_ep_steps
        self.rewards = rewards
        self.transition_probabilities = transition_probabilities
        self.initial_state = initial_state
        self.final_state = final_state

        self.num_states = transition_probabilities.shape[0]
        self.num_actions = transition_probabilities.shape[1]

        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)

    def reset(self, seed=None):
        self.step_cnt = 0
        self.state = self.initial_state
        return self.state, {}

    def is_done(self, state):
        return state == self.final_state

    def reward(self, state: int, action: int, next_state: int):
        return self.rewards[state][action][next_state]

    def step(self, action: int):
        old_state = self.state

        self.state = np.random.choice(self.num_states, p=self.transition_probabilities[self.state][action])
        self.step_cnt += 1

        truncated = self.step_cnt >= self.max_ep_steps

        return self.state, self.reward(old_state, action, self.state), self.is_done(self.state), truncated, {}

    def next_state_distribution(self, observation: int, action: int) -> np.ndarray:
        return self.transition_probabilities[observation][action]

    def render(self):
        print(f"Current state: {self.state}")

    def draw_mdp(self, filepath: Path, format: Literal["png", "svg"] = "png"):
        draw_mdp(self.transition_probabilities, self.rewards, filepath, format)


class RandomMDP(MatrixMDP):
    def __init__(
        self,
        max_ep_steps: int,
        num_states: int,
        num_actions: int,
        rewards: np.ndarray,
        initial_state: int,
        final_state: int = -1,
        randomize_on_reset: bool = True,
    ):
        super().__init__(max_ep_steps, self._random_transition_matrix(num_states, num_actions), rewards, initial_state, final_state)
        self.randomize_on_reset = randomize_on_reset

    def _random_transition_matrix(self, num_states: int, num_actions: int) -> np.ndarray:
        uniform_simplex = torch.distributions.Dirichlet(torch.ones(num_states))
        return uniform_simplex.sample((num_states, num_actions)).numpy()

    def reset(self, seed=None):
        self.step_cnt = 0
        self.state = self.initial_state
        if self.randomize_on_reset:
            self.transition_probabilities = self._random_transition_matrix(self.num_states, self.num_actions)
        return self.state, {}


def transition_probabilities_from_world_model(world_model, observation_dim, action_dim):
    transition_probabilities = []
    for s in range(observation_dim):
        transition_probabilities.append([])
        for a in range(action_dim):
            next_state_probabilities, _ = world_model.predict((s, a))
            next_state_probabilities = F.softmax(torch.from_numpy(next_state_probabilities), dim=-1)
            transition_probabilities[-1].append(np.array(next_state_probabilities))
    return np.array(transition_probabilities)


def draw_mdp(transition_probabilities: np.ndarray, rewards: np.ndarray, filepath: Path, format: Literal["png", "svg"] = "png"):
    # Contruct edges and combine multiple labels for the same edge to one
    edges = defaultdict(list)
    for u in range(transition_probabilities.shape[0]):
        for a in range(transition_probabilities.shape[1]):
            for v in range(transition_probabilities.shape[2]):
                if transition_probabilities[u][a][v] > 0:
                    edges[(u, v)].append(f"a{a} | {transition_probabilities[u][a][v]:.2f} | {rewards[u][a][v]:.2f}")

    mdp = pgv.AGraph(directed=True)

    for (u, v), labels in edges.items():
        # Need to add edges like this to be able to provide a label
        mdp.add_edge(f"s{u}", f"s{v}", label="\n".join(labels))

    mdp.layout()
    # Create parent directories if they don't exists yet
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    mdp.draw(filepath, prog="dot", format=format)


class LearnableWorldModel(WorldModel):
    def __init__(
        self,
        learnable_model: BasePolicy,
        num_states: int,
        num_actions: int,
        max_ep_steps: int,
        rewards: np.ndarray,
        initial_state: int,
        final_state: int = -1,
    ):
        self.learnable_model = learnable_model
        self.num_states = num_states
        self.num_actions = num_actions
        self.max_ep_steps = max_ep_steps
        self.rewards = rewards
        self.initial_state = initial_state
        self.final_state = final_state

        self.max_ep_steps = max_ep_steps

        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)

    def reset(self, seed=None):
        self.step_cnt = 0
        self.state = self.initial_state
        return self.state, {}

    def is_done(self, state):
        return state == self.final_state

    def reward(self, state: int, action: int, next_state: int):
        return self.rewards[state][action][next_state]

    def step(self, action: int):
        old_state = self.state

        self.state = np.random.choice(self.num_states, p=self.next_state_distribution(self.state, action))
        self.step_cnt += 1

        truncated = self.step_cnt >= self.max_ep_steps

        return self.state, self.reward(old_state, action, self.state), self.is_done(self.state), truncated, {}

    def next_state_distribution(self, observation: int, action: int) -> np.ndarray:
        next_state_predictions, _ = self.learnable_model.predict((observation, action))
        return F.softmax(torch.from_numpy(next_state_predictions), dim=-1).numpy()

    def render(self):
        print(f"Current state: {self.state}")


class LearnableWorldModel(WorldModel):
    def __init__(
        self,
        learnable_model: BasePolicy,
        num_states: int,
        num_actions: int,
        max_ep_steps: int,
        rewards: np.ndarray,
        initial_state: int,
        final_state: int = -1,
    ):
        self.learnable_model = learnable_model
        self.num_states = num_states
        self.num_actions = num_actions
        self.max_ep_steps = max_ep_steps
        self.rewards = rewards
        self.initial_state = initial_state
        self.final_state = final_state

        self.max_ep_steps = max_ep_steps

        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)

    def reset(self, seed=None):
        self.step_cnt = 0
        self.state = self.initial_state
        return self.state, {}

    def is_done(self, state):
        return state == self.final_state

    def reward(self, state: int, action: int, next_state: int):
        return self.rewards[state][action][next_state]

    def step(self, action: int):
        old_state = self.state

        self.state = np.random.choice(self.num_states, p=self.next_state_distribution(self.state, action))
        self.step_cnt += 1

        truncated = self.step_cnt >= self.max_ep_steps

        return self.state, self.reward(old_state, action, self.state), self.is_done(self.state), truncated, {}

    def next_state_distribution(self, observation: int, action: int) -> np.ndarray:
        next_state_predictions, _ = self.learnable_model.predict((observation, action))
        return F.softmax(torch.from_numpy(next_state_predictions), dim=-1).numpy()

    def render(self):
        print(f"Current state: {self.state}")