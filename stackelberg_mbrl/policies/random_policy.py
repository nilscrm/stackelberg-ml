import torch
import numpy as np
from stackelberg_mbrl.policies.policy import APolicy


class RandomPolicy(APolicy):
    """ Policy that can be randomly set"""
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        seed = None
    ):
        self.num_states = num_states
        self.num_actions = num_actions

        self.randomize(seed)

    def randomize(self, seed=None):
        uniform_simplex = torch.distributions.Dirichlet(torch.ones(self.num_actions))
        return uniform_simplex.sample([self.num_states]).numpy()