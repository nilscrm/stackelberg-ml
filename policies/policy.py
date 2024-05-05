from abc import ABC, abstractmethod

import torch
from util.tensor_util import tensorize_array_inputs

class APolicy(ABC):
    
    @property
    def trainable_params(self):
        None

    @abstractmethod
    def next_action_distribution(self, observation):
        pass

    @abstractmethod
    def sample_next_action(self, observation):
        pass

    @tensorize_array_inputs
    def sample_next_actions(self, observations):
        """ Sample the next actions for multiple observations """
        return torch.concatenate([self.sample_next_action(observations[i]).unsqueeze(0) for i in range(observations.shape[0])])

    @abstractmethod
    def log_likelihood(self, observations, groundtruth_actions):
        """ Predicts the actions that will be taken for some observations and computes the log likelihood of them given the groundtruth actions """
        pass

    @abstractmethod
    def kl_divergence(self, observations, old_actions):
        """ Predicts the actions that will be taken for some observations and computes the kl divergence KL(new_actions||old_actions)"""
        pass