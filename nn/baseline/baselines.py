# TODO: do we need to condition baseline too???
from abc import abstractmethod

import numpy as np
import torch
from util.optimization import fit as fit_model
from util.tensor_util import tensorize_array_inputs

from nn.mlp import MLP

class ABaseline:
    """ Predict the expected returns from observations in trajectories """
    @abstractmethod
    def fit(self, observations: torch.Tensor, returns: torch.Tensor, return_errors: bool = False):
        pass

    @abstractmethod
    def predict_expected_returns(self, observations: torch.Tensor) -> torch.Tensor:
        pass   


class BaselineMLP(ABaseline):
    # TODO: in mjrl, they appended 4 small numbers (powers) to the end of the featurevector, not sure if we need this too
    """ Learns to predict the expected return for trajectories using a MLP """
    def __init__(self, input_dim: int, hidden_sizes=(128, 128), learn_rate=1e-3, reg_coef=0.0,
                 batch_size=64, epochs=1):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs

        self.mlp = MLP(input_dim, hidden_sizes, 1)

        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=learn_rate, weight_decay=reg_coef)
        self.loss_function = torch.nn.MSELoss()


    @tensorize_array_inputs
    def fit(self, observations, returns, return_errors=False):
        if return_errors:
            errors = returns - self.mlp(observations)
            error_before = torch.sum(errors**2)/(torch.sum(returns**2) + 1e-8)

            fit_model(self.mlp, observations, returns, self.optimizer, self.loss_function, self.batch_size, self.epochs)

            errors = returns - self.mlp(observations)
            error_after = torch.sum(errors**2)/(torch.sum(returns**2) + 1e-8)
            return error_before.detach().numpy(), error_after.detach().numpy()
        else:
            fit_model(self.mlp, observations, returns, self.optimizer, self.loss_function, self.batch_size, self.epochs)
            
    @tensorize_array_inputs
    def predict_expected_returns(self, observations: np.ndarray):
        prediction = self.mlp(observations).flatten()
        return prediction
