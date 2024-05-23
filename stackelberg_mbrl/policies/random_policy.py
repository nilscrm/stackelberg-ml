import torch
import numpy as np
from stackelberg_mbrl.policies.policy import APolicy


class RandomPolicy(APolicy):
    """ Policy that can be randomly set"""
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        context_size: int,
        seed = None
    ):
        self.num_states = num_states
        self._num_actions = num_actions
        self.context_size = context_size

        self.randomize(seed)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def randomize(self, seed=None):
        uniform_simplex = torch.distributions.Dirichlet(torch.ones(self.num_actions))
        self.action_probabilities = uniform_simplex.sample([self.num_states]).numpy()
    
    def next_action_distribution(self, observation: int | np.ndarray) -> np.ndarray:
        if isinstance(observation, int):
            state_idx = observation
        else:
            # observation is expected to be of format (context, state_one_hot)
            state_idx = np.argmax(observation[self.context_size:]).item()
        return self.action_probabilities[state_idx]

    def sample_next_action(self, observation: np.ndarray) -> int:
        action_probs = self.next_action_distribution(observation)
        return torch.multinomial(torch.tensor(action_probs), num_samples=1).numpy().item()