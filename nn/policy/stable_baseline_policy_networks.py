import gymnasium.spaces as spaces

import torch
from util.tensor_util import tensorize_array_inputs

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SB3ContextualizedFeatureExtractor(BaseFeaturesExtractor):
    """ Pseudo-feature extractor, that appends a context to each observation """
    def __init__(self, observation_space: spaces.Discrete, context_size: int):
        out_dim = observation_space.n + context_size
        super().__init__(observation_space, out_dim)

        self.context_size = context_size
        self.context = torch.zeros(context_size)

    @tensorize_array_inputs
    def set_context(self, context: torch.Tensor):
        self.context = context

    def forward(self, observations: torch.Tensor):
        if observations.ndim == 2 and observations.shape[0] == 1:
            return torch.concatenate([observations, self.context.tile(observations.shape[0],1)], dim=-1)
        elif observations.ndim == 3 and observations.shape[1] == 1:
            return torch.concatenate([observations, self.context.tile(observations.shape[0],1).unsqueeze(1)], dim=-1)