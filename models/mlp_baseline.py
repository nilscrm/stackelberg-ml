import numpy as np
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
from util.optimize_model import fit_data
from util.tensor_utils import tensorize

import pickle

class MLPBaseline:
    def __init__(self, env_spec, inp_dim=None, inp='obs', learn_rate=1e-3, reg_coef=0.0,
                 batch_size=64, epochs=1, device='cpu', hidden_sizes=(128, 128), *args, **kwargs):
        self.n = inp_dim if inp_dim is not None else env_spec.observation_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_coef = reg_coef
        self.device = device
        self.inp = inp
        self.hidden_sizes = hidden_sizes

        self.model = nn.Sequential()
        layer_sizes = (self.n + 4, ) + hidden_sizes + (1, )
        for i in range(len(layer_sizes) - 1):
            layer_id = 'fc_' + str(i)
            relu_id = 'relu_' + str(i)
            self.model.add_module(layer_id, nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i != len(layer_sizes) - 2:
                self.model.add_module(relu_id, nn.ReLU())

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate, weight_decay=reg_coef)
        self.loss_function = torch.nn.MSELoss()

    def _features(self, paths):
        if self.inp == 'env_features':
            o = np.concatenate([path["env_infos"]["env_features"][0] for path in paths])
        else:
            o = np.concatenate([path["observations"] for path in paths])
        o = np.clip(o, -10, 10)/10.0
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        N, n = o.shape
        num_feat = int( n + 4 )            # linear + time till pow 4
        feat_mat =  np.ones((N, num_feat)) # memory allocation

        # linear features
        feat_mat[:,:n] = o

        k = 0  # start from this row
        for i in range(len(paths)):
            l = len(paths[i]["rewards"])
            al = np.arange(l)/1000.0
            for j in range(4):
                feat_mat[k:k+l, -4+j] = al**(j+1)
            k += l
        return feat_mat


    def fit(self, paths, return_errors=False):

        featmat = self._features(paths)
        returns = np.concatenate([path["returns"] for path in paths]).reshape(-1, 1)
        featmat = featmat.astype('float32')
        returns = returns.astype('float32')
        num_samples = returns.shape[0]

        # Make variables with the above data
        featmat_var = tensorize(featmat, device=self.device)
        returns_var = tensorize(returns, device=self.device)

        if return_errors:
            predictions = self.model(featmat_var).to('cpu').data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_before = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)

        epoch_losses = fit_data(self.model, featmat_var, returns_var, self.optimizer,
                                self.loss_function, self.batch_size, self.epochs)

        if return_errors:
            predictions = self.model(featmat_var).to('cpu').data.numpy().ravel()
            errors = returns.ravel() - predictions
            error_after = np.sum(errors**2)/(np.sum(returns**2) + 1e-8)
            return error_before, error_after

    def predict(self, path):
        featmat = self._features([path]).astype('float32')
        featmat_var = tensorize(featmat, device=self.device)
        prediction = self.model(featmat_var).to('cpu').data.numpy().ravel()
        return prediction
