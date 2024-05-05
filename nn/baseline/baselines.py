# TODO: do we need to condition baseline too???
from abc import abstractmethod

import numpy as np
import torch
from util.optimization import fit
from util.tensor_util import tensorize_array_inputs

from nn.mlp import MLP

class ABaseline:
    """ Predict the expected returns from observations in trajectories """
    @abstractmethod
    def fit(self, observations: np.ndarray, returns: np.ndarray, return_errors: bool = False):
        pass

    @abstractmethod
    def predict_expected_returns(self, observations: np.ndarray):
        pass

class AverageBaseline(ABaseline):
    def __init__(self):
        self.avg_return = 0.0

    @tensorize_array_inputs
    def fit(self, observations, returns, return_errors=False):
        if return_errors:
            errors = returns - self.avg_return    
            error_before = torch.sum(errors**2)/(torch.sum(returns**2) + 1e-8)

            self.avg_return = returns.mean()
            
            errors = returns - self.avg_return    
            error_after = torch.sum(errors**2)/(torch.sum(returns**2) + 1e-8)
            
            return error_before.numpy(), error_after.numpy()


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

            fit(self.mlp, observations, returns, self.optimizer, self.loss_function, self.batch_size, self.epochs)

            errors = returns - self.mlp(observations)
            error_after = torch.sum(errors**2)/(torch.sum(returns**2) + 1e-8)
            return error_before.numpy(), error_after.numpy()
        else:
            fit(self.mlp, observations, returns, self.optimizer, self.loss_function, self.batch_size, self.epochs)
            
    @tensorize_array_inputs
    def predict_expected_returns(self, observations: np.ndarray):
        prediction = self.mlp(observations).numpy().ravel()
        return prediction
