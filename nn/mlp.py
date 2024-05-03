from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: List[int], output_dim: int, nonlinearity = torch.relu):
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_sizes = [input_dim, *hidden_size, output_dim]
        
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                        for i in range(len(self.layer_sizes)-1)])
        self.nonlinearity = nonlinearity

    def forward(self, x):
        out = x
        for i in range(len(self.fc_layers) - 1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        return self.fc_layers[-1](out)
