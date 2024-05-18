# NOTE: mjrl impl offered residual connection and rescaling but that does not make sense for discrete states

import torch
import torch.nn as nn
import torch.nn.functional as F

from stackelberg_mbrl.nn.mlp import MLP

class DynamicsNetMLP(nn.Module):
    """ Simple fully connected dynamics model that is parameterized as f_theta(s,a) -> s """
    def __init__(self, state_dim, act_dim, hidden_sizes=(64,64), nonlinearity = torch.relu):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.mlp = MLP(
            input_dim=state_dim + act_dim,
            hidden_sizes=hidden_sizes,
            output_dim=state_dim,
            nonlinearity=nonlinearity
        )

    def forward(self, s, a):
        assert s.dim() == a.dim(), "State and action inputs should be of the same size"
        return F.softmax(self.mlp.forward(torch.cat([s, a], -1)))
