from abc import ABC
from collections import defaultdict
import gymnasium
import numpy as np
from pathlib import Path
import pygraphviz as pgv
import torch
from torch.nn.functional import softmax
from typing import Literal

class AEnv(ABC, gymnasium.Env):
    def __init__(self) -> None:
        gymnasium.Env.__init__(self)

    @property
    def max_episode_steps(self):
        pass

    @property
    def observation_dim(self):
        pass

    @property
    def action_dim(self):
        pass


class DiscreteEnv(AEnv):
    """
        Adds convenience functions to a gym.Env with discrete action and observation space
    """
    def __init__(self, max_episode_steps) -> None:
        super().__init__()

        self._max_episode_steps = max_episode_steps

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def observation_dim(self):
        return self.observation_space.n

    @property
    def action_dim(self):
        return self.action_space.n
    
def transition_probabilities_from_world_model(world_model, observation_dim, action_dim):
    transition_probabilities = []
    for s in range(observation_dim):
        transition_probabilities.append([])
        for a in range(action_dim):
            next_state_probabilities, _ = world_model.predict((s, a))
            next_state_probabilities = softmax(torch.from_numpy(next_state_probabilities))
            transition_probabilities[-1].append(np.array(next_state_probabilities))
    return np.array(transition_probabilities)

    
def draw_mdp(transition_probabilities: np.ndarray, rewards: np.ndarray, filepath: Path, format: Literal['png', 'svg'] = 'png'):
    # Contruct edges and combine multiple labels for the same edge to one
    edges = defaultdict(list)
    for u in range(transition_probabilities.shape[0]):
        for a in range(transition_probabilities.shape[1]):
            for v in range(transition_probabilities.shape[2]):
                if transition_probabilities[u][a][v] > 0:
                    edges[(u, v)].append(f"a{a} | {transition_probabilities[u][a][v]:.2f} | {rewards[u][a][v]}")

    mdp = pgv.AGraph(directed=True)

    for (u, v), labels in edges.items():
        # Need to add edges like this to be able to provide a label
        mdp.add_edge(f"s{u}", f"s{v}", label="\n".join(labels))

    mdp.layout()
    # Create parent directories if they don't exists yet
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    mdp.draw(filepath, prog='dot', format=format)
