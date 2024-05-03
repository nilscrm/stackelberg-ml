# NOTE: mjrl used a gaussian MLP to model the policy, but that makes little sense in a discrete setting

import numpy as np
import torch
import torch.nn as nn
from nn.mlp import MLP
from policies.policy import APolicy
from util.tensor_util import tensorize_array_inputs
import torch.nn.functional as F

class PolicyFC(nn.Module, APolicy):
    """ 
        MLP that parameterizes a policy.
        
        Can receive an extended input (e.g. to condition on the world model), by setting context_size > 0
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_sizes=(64,64), context_size: int=0):
        nn.Module.__init__(self)
        APolicy.__init__(self)

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.context_size = context_size

        self.mlp = MLP(
            input_dim=observation_dim + context_size,
            hidden_size=hidden_sizes,
            output_dim=action_dim,
            nonlinearity=torch.tanh
        )
        # TODO: mjrl rescaled the last layer parameters by 1/100, not sure if we need this too

        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]

    @property
    def trainable_params(self):
        return list(self.parameters())

    @tensorize_array_inputs
    def next_action_distribution(self, observation):
        return F.softmax(self.mlp(observation))

    @tensorize_array_inputs
    def sample_next_action(self, observation):
        action = self.next_action_distribution(observation.flatten())
        return F.one_hot(torch.multinomial(action), num_classes=self.action_dim)

    @tensorize_array_inputs
    def log_likelihood(self, observations, groundtruth_actions):
        predicted_actions = self.next_action_distribution(observations)
        # TODO: check this is correct (should be the sum over the log of the predicted values of the groundtruth label, I'm assuming groundtruth_actions are one-hot-encoded)
        return -torch.sum(groundtruth_actions * torch.log(predicted_actions + 1e-8), dim=1).numpy()

    def kl_divergence(self, observations, old_actions):
        predicted_actions = self.next_action_distribution(observations)
        # TODO: check it is correct that we use the new actions as the groundtruth
        return F.kl_div(old_actions, predicted_actions)

    def get_param_values(self):
        params = torch.cat([p.contiguous().view(-1).data for p in self.parameters()])
        return params.clone()

    def set_param_values(self, new_params):
        current_idx = 0
        for idx, param in enumerate(self.parameters()):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            current_idx += self.param_sizes[idx]