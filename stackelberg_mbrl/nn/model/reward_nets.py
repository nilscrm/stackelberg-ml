
import torch
import torch.nn as nn

from stackelberg_mbrl.nn.mlp import MLP

class RewardNetMLP(nn.Module):
    """ Simple fully connected reward network that is parameterized as f_theta(s, a, s_next) """
    def __init__(self, state_dim, act_dim, hidden_size=(64,64), nonlinearity = torch.relu):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.mlp = MLP(
            input_dim=state_dim + act_dim + state_dim,
            hidden_sizes=hidden_size,
            output_dim=1,
            nonlinearity=nonlinearity
        )
        
    def forward(self, s, a, s_next):
        assert s.dim() == a.dim(), "State and action inputs should be of the same size"
        return self.mlp(torch.cat([s, a, s_next], -1))
