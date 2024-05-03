import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F

from models.nn_dynamics import AWorldModel

class RandomWorldModel(AWorldModel):
    def __init__(self, state_dim, act_dim,
                 learn_reward=False,
                 seed=123,
                 device='cpu'):

        self.state_dim, self.act_dim = state_dim, act_dim
        self.device, self.learn_reward = device, learn_reward
        self.seed = seed

        self.uniform_simplex = torch.distributions.Dirichlet(torch.ones(self.state_dim))

        self.randomize()

    def randomize(self):
        self.transition_probabilities = self.uniform_simplex.sample((self.state_dim, self.act_dim))
    
        if self.learn_reward:
            self.rewards = 
            
        else:
            self.reward_net, self.reward_opt, self.reward_loss = None, None, None
    
    def next_state_distribution(self, s, a):
        state = np.argmax(s).item()
        action = np.argmax(a).item()

        return self.transition_probabilities[state,action]
    
    def sample_next_state(self, s, a):
        return F.one_hot(torch.multinomial(self.next_state_distribution(s, a), num_samples=1), num_classes=self.state_dim)

    def forward(self, s, a):
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
        if type(a) == np.ndarray:
            a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        return self.dynamics_net.forward(s, a)

    def predict(self, s, a):
        s = torch.from_numpy(s).float()
        a = torch.from_numpy(a).float()
        s = s.to(self.device)
        a = a.to(self.device)
        s_next = self.dynamics_net.forward(s, a)
        s_next = s_next.to('cpu').data.numpy()
        return s_next

    def reward(self, s, a):
        if not self.learn_reward:
            print("Reward model is not learned. Use the reward function from env.")
            return None
        else:
            if type(s) == np.ndarray:
                s = torch.from_numpy(s).float()
            if type(a) == np.ndarray:
                a = torch.from_numpy(a).float()
            s = s.to(self.device)
            a = a.to(self.device)
            sp = self.dynamics_net.forward(s, a).detach().clone()
            return self.reward_net.forward(s, a, sp)

    def compute_loss(self, s, a, s_next):
        # Intended for logging use only, not for loss computation
        sp = self.forward(s, a)
        s_next = torch.from_numpy(s_next).float() if type(s_next) == np.ndarray else s_next
        s_next = s_next.to(self.device)
        loss = self.dynamics_loss(sp, s_next)
        return loss.to('cpu').data.numpy()

    def fit_dynamics(self, s, a, sp, fit_mb_size, fit_epochs, max_steps=1e4, 
                     set_transformations=True, *args, **kwargs):
        # move data to correct devices
        assert type(s) == type(a) == type(sp)
        assert s.shape[0] == a.shape[0] == sp.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            sp = torch.from_numpy(sp).float()
        s = s.to(self.device); a = a.to(self.device); sp = sp.to(self.device)
       
        # set network transformations
        if set_transformations:
            s_shift, a_shift = torch.mean(s, dim=0), torch.mean(a, dim=0)
            s_scale, a_scale = torch.mean(torch.abs(s - s_shift), dim=0), torch.mean(torch.abs(a - a_shift), dim=0)
            out_shift = torch.mean(sp-s, dim=0) if self.dynamics_net.residual else torch.mean(sp, dim=0)
            out_scale = torch.mean(torch.abs(sp-s-out_shift), dim=0) if self.dynamics_net.residual else torch.mean(torch.abs(sp-out_shift), dim=0)
            self.dynamics_net.set_transformations(s_shift, s_scale, a_shift, a_scale, out_shift, out_scale)

        X = (s, a)
        Y = sp
        # prepare dataf for learning
        # if self.dynamics_net.residual:  
        #     X = (s, a) ; Y = (sp - s - out_shift) / (out_scale + 1e-8)
        # else:
        #     X = (s, a) ; Y = (sp - out_shift) / (out_scale + 1e-8)
        # disable output transformations to learn in the transformed space
        self.dynamics_net._apply_out_transforms = False
        return_vals =  fit_model(self.dynamics_net, X, Y, self.dynamics_opt, self.dynamics_loss,
                                 fit_mb_size, fit_epochs, max_steps=max_steps)
        self.dynamics_net._apply_out_transforms = True
        return return_vals

    def fit_reward(self, s, a, r, fit_mb_size, fit_epochs, max_steps=1e4, 
                   set_transformations=True, *args, **kwargs):
        if not self.learn_reward:
            print("Reward model was not initialized to be learnable. Use the reward function from env.")
            return None

        # move data to correct devices
        assert type(s) == type(a) == type(r)
        assert len(r.shape) == 2 and r.shape[1] == 1  # r should be a 2D tensor, i.e. shape (N, 1)
        assert s.shape[0] == a.shape[0] == r.shape[0]
        if type(s) == np.ndarray:
            s = torch.from_numpy(s).float()
            a = torch.from_numpy(a).float()
            r = torch.from_numpy(r).float()
        s = s.to(self.device); a = a.to(self.device); r = r.to(self.device)
       
        # set network transformations
        if set_transformations:
            s_shift, a_shift = torch.mean(s, dim=0), torch.mean(a, dim=0)
            s_scale, a_scale = torch.mean(torch.abs(s-s_shift), dim=0), torch.mean(torch.abs(a-a_shift), dim=0)
            r_shift = torch.mean(r, dim=0)
            r_scale = torch.mean(torch.abs(r-r_shift), dim=0)
            self.reward_net.set_transformations(s_shift, s_scale, a_shift, a_scale, r_shift, r_scale)

        # get next state prediction
        sp = self.dynamics_net.forward(s, a).detach().clone()

        # call the generic fit function
        X = (s, a, sp) ; Y = r
        return fit_model(self.reward_net, X, Y, self.reward_opt, self.reward_loss,
                         fit_mb_size, fit_epochs, max_steps=max_steps)

    def compute_path_rewards(self, paths):
        # paths has two keys: observations and actions
        # paths["observations"] : (num_traj, horizon, obs_dim)
        # paths["rewards"] should have shape (num_traj, horizon)
        if not self.learn_reward: 
            print("Reward model is not learned. Use the reward function from env.")
            return None
        s, a = paths['observations'], paths['actions']
        num_traj, horizon, s_dim = s.shape
        a_dim = a.shape[-1]
        s = s.reshape(-1, s_dim)
        a = a.reshape(-1, a_dim)
        r = self.reward(s, a)
        r = r.to('cpu').data.numpy().reshape(num_traj, horizon)
        paths['rewards'] = r