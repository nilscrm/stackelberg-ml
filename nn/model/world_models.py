from typing import Callable, Optional

from abc import abstractmethod
import numpy as np
import torch
from torch.nn import functional as F

from envs.env_util import DiscreteEnv
from nn.model.dynamics_nets import DynamicsNetMLP
from nn.model.reward_nets import RewardNetMLP

from util.optimization import fit_tuple
from util.tensor_util import extract_one_hot_index_inputs, tensorize, tensorize_array_inputs

class AWorldModel:
    @property
    def action_dim(self):
        pass

    @property
    def observation_dim(self):
        pass

    @abstractmethod
    def sample_initial_state(self) -> np.ndarray:
        """ Sample one state from the distribution over initial states """
        pass

    @abstractmethod
    def next_state_distribution(self, s, a) -> np.ndarray:
        """ Get the probabilities of ending up in each state, given that action a is taken in state s """
        pass

    @abstractmethod
    def sample_next_state(self, s, a) -> np.ndarray:
        """ Sample one state from the distribution over next states """
        pass

    @abstractmethod
    def reward(self, s, a, s_next) -> float:
        pass

    @abstractmethod
    def is_done(self, s) -> bool:
        pass

class WorldModel(AWorldModel):
    """ 
        Model of an environment, consisting of 
        - Dynamics Model: Models the probabilities of transitioning to a new state given a state-action pair
        - Reward Model: If no reward function is provided, the model learns the rewards for each state-action pair
    """
    def __init__(self, state_dim, act_dim,
                 hidden_sizes=(64,64),
                 activation=torch.relu,
                 reward_func=None, # reward will be learned if no reward function is provided
                 fit_lr=1e-3,
                 fit_weight_decay=0.0):

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.reward_function = reward_func
        
        # construct the dynamics model
        self.dynamics_net = DynamicsNetMLP(state_dim, act_dim, hidden_sizes, nonlinearity=activation)

        self.dynamics_opt = torch.optim.Adam(self.dynamics_net.parameters(), lr=fit_lr, weight_decay=fit_weight_decay)
        self.dynamics_loss = torch.nn.MSELoss()

        # construct the reward model (if reward is learned)
        if self.reward_function is None:
            # small network for reward is sufficient if we augment the inputs with next state predictions
            self.reward_net = RewardNetMLP(state_dim, act_dim, hidden_size=(100, 100), nonlinearity=activation)

            self.reward_opt = torch.optim.Adam(self.reward_net.parameters(), lr=fit_lr, weight_decay=fit_weight_decay)
            self.reward_loss = torch.nn.MSELoss()
        else:
            self.reward_net, self.reward_opt, self.reward_loss = None, None, None

    @property
    def action_dim(self):
        return self.act_dim

    @property
    def observation_dim(self):
        return self.state_dim

    @tensorize_array_inputs
    def next_state_distribution(self, s, a):
        out = self.dynamics_net.forward(s, a)
        return F.softmax(out, dim=-1)
    
    @tensorize_array_inputs
    def sample_next_state(self, s, a):
        state_idx = torch.multinomial(self.next_state_distribution(s, a), num_samples=1)
        return F.one_hot(state_idx, num_classes=self.state_dim)[0] # convert to non-batched

    @tensorize_array_inputs
    def reward(self, s, a, s_next):
        if self.reward_function:
            return self.reward_function(s,a)
        else:
            return self.reward_net.forward(s, a, s_next)

    @tensorize_array_inputs
    def fit_dynamics(self, s, a, s_next, fit_mb_size, fit_epochs, max_steps=1e4):
        assert s.shape[0] == a.shape[0] == s_next.shape[0]

        X = (s, a)
        Y = s_next
        return fit_tuple(self.dynamics_net, X, Y, self.dynamics_opt, self.dynamics_loss, 
                         fit_mb_size, fit_epochs, max_steps=max_steps)

    @tensorize_array_inputs
    def fit_reward(self, s, a, s_next, r, fit_mb_size, fit_epochs, max_steps=1e4):
        assert self.reward_function is None, "Reward model was not initialized to be learnable. Use the reward function from env."

        assert len(r.shape) == 2 and r.shape[1] == 1  # r should be a 2D tensor, i.e. shape (N, 1)
        assert s.shape[0] == a.shape[0] == r.shape[0] == s_next.shape[0]

        X = (s, a, s_next)
        Y = r
        return fit_tuple(self.reward_net, X, Y, self.reward_opt, self.reward_loss,
                         fit_mb_size, fit_epochs, max_steps=max_steps)




class RandomDiscreteModel(AWorldModel):
    """ Random model of the world with a discrete action and observation space """
    def __init__(self, template: DiscreteEnv, init_state_probs: np.ndarray, 
                 termination_func: Callable, reward_func: Optional[Callable] = None,
                 min_reward: float = 0.0, max_reward: float = 1.0):
        self.state_space = template.observation_space
        self.act_space = template.action_space
        self.state_dim = template.observation_dim
        self.act_dim = template.action_dim

        self.termination_func = termination_func
        self.reward_func = reward_func
        self.min_reward = min_reward
        self.max_reward = max_reward

        self.init_state_sampler = lambda : torch.multinomial(tensorize(init_state_probs), num_samples=1)[0] # convert to non-batched
        self.uniform_simplex = torch.distributions.Dirichlet(torch.ones(self.state_dim))

        self.randomize()

    def randomize(self):
        self.transition_probabilities = self.uniform_simplex.sample((self.state_dim, self.act_dim))
    
        if self.reward_func is None:
            self.rewards = (torch.rand((self.state_dim, self.act_dim, self.state_dim)) + self.min_reward) * (self.max_reward - self.min_reward)

    @property
    def action_dim(self):
        return self.act_dim

    @property
    def observation_dim(self):
        return self.state_dim

    @property
    def action_space(self):
        return self.action_space

    @property
    def observation_space(self):
        return self.observation_space

    def sample_initial_state(self):
        state_idx = self.init_state_sampler()
        return F.one_hot(state_idx, num_classes=self.state_dim)

    @extract_one_hot_index_inputs
    def next_state_distribution(self, s, a):
        return self.transition_probabilities[s,a]

    def sample_next_state(self, s, a):
        state_idx = torch.multinomial(self.next_state_distribution(s, a), num_samples=1)[0] # convert to non-batched
        return F.one_hot(state_idx, num_classes=self.state_dim)

    def reward(self, s, a, s_next):
        if self.reward_func is None:
            return self._apply_reward_func(s, a, s_next)
        else:
            return self.reward_func(s,a,s_next)
    
    @extract_one_hot_index_inputs
    def _apply_reward_func(self, s, a, s_next):
        return self.rewards[s,a,s_next]

    def is_done(self, s):
        return self.termination_func(s)