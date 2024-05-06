import numpy as np
import torch

from policies.policy import ATrainablePolicy
from util.tensor_util import tensorize_array_inputs

class ContextualizedPolicy(ATrainablePolicy):
    """ Basically a wrapper that appends a context to the observation when choosing an action """

    def __init__(self, policy: ATrainablePolicy, initial_context: np.ndarray):
        super().__init__()
        self.policy = policy
        self.context = initial_context

    def set_context(self, context: np.ndarray):
        self.context = context

    @property
    def trainable_params(self):
        return self.policy.trainable_params
    
    @tensorize_array_inputs
    def next_action_distribution(self, observation):
        return self.policy.next_action_distribution(torch.concatenate([observation, self.context], axis=0))

    @tensorize_array_inputs
    def sample_next_action(self, observation):
        return self.policy.sample_next_action(torch.concatenate([observation, self.context], axis=0))
    
    @tensorize_array_inputs
    def log_likelihood(self, observations, groundtruth_actions):
        return self.policy.log_likelihood(self._append_context(observations), groundtruth_actions)

    @tensorize_array_inputs
    def kl_divergence(self, observations, old_actions):
        return self.policy.kl_divergence(self._append_context(observations), old_actions)
    
    def _append_context(self, observations):
        rows,cols = observations.shape
        tiled_vector = np.tile(self.context, (rows, 1))
        observation_with_context = np.zeros((rows, cols + len(self.context)))
        observation_with_context[:,:cols] = observations
        observation_with_context[:,cols:] = tiled_vector
        return observation_with_context
    
    def get_param_values(self):
        return self.policy.get_param_values()

    def set_param_values(self, new_params):
        self.policy.set_param_values(new_params)