from abc import abstractmethod
import numpy as np
import torch
from torch.nn import functional as F

from envs.gym_env import GymEnv
from nn.model.dynamics_nets import DynamicsNetFC
from nn.model.reward_nets import RewardNetFC

from nn.model.training_util import fit_model
from util.tensor_util import extract_one_hot_index_inputs, tensorize_array_inputs


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
                 hidden_size=(64,64),
                 fit_lr=1e-3,
                 fit_wd=0.0,
                 activation=torch.relu,
                 reward_function=None, # reward will be learned if no reward function is provided
                 residual=True):

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.reward_function = reward_function
        
        # construct the dynamics model
        self.dynamics_net = DynamicsNetFC(state_dim, act_dim, hidden_size, residual=residual, nonlinearity=activation)

        self.dynamics_opt = torch.optim.Adam(self.dynamics_net.parameters(), lr=fit_lr, weight_decay=fit_wd)
        self.dynamics_loss = torch.nn.MSELoss()

        # construct the reward model (if reward is learned)
        if self.reward_function is None:
            # small network for reward is sufficient if we augment the inputs with next state predictions
            self.reward_net = RewardNetFC(state_dim, act_dim, hidden_size=(100, 100), nonlinearity=activation)

            self.reward_opt = torch.optim.Adam(self.reward_net.parameters(), lr=fit_lr, weight_decay=fit_wd)
            self.reward_loss = torch.nn.MSELoss()
        else:
            self.reward_net, self.reward_opt, self.reward_loss = None, None, None

    @tensorize_array_inputs
    def next_state_distribution(self, s, a):
        out = self.dynamics_net.forward(s, a)
        return F.softmax(out, dim=-1)
    
    @tensorize_array_inputs
    def sample_next_state(self, s, a):
        state_idx = torch.multinomial(self.next_state_distribution(s, a), num_samples=1)
        return F.one_hot(state_idx, num_classes=self.state_dim)

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
        return fit_model(self.dynamics_net, X, Y, self.dynamics_opt, self.dynamics_loss, 
                         fit_mb_size, fit_epochs, max_steps=max_steps)

    @tensorize_array_inputs
    def fit_reward(self, s, a, s_next, r, fit_mb_size, fit_epochs, max_steps=1e4):
        assert self.reward_function is None, "Reward model was not initialized to be learnable. Use the reward function from env."

        assert len(r.shape) == 2 and r.shape[1] == 1  # r should be a 2D tensor, i.e. shape (N, 1)
        assert s.shape[0] == a.shape[0] == r.shape[0] == s_next.shape[0]

        X = (s, a, s_next)
        Y = r
        return fit_model(self.reward_net, X, Y, self.reward_opt, self.reward_loss,
                         fit_mb_size, fit_epochs, max_steps=max_steps)




class RandomDiscreteModel(AWorldModel):
    def __init__(self, template: GymEnv, reward_func: callable | None = None, 
                 min_reward: float = 0.0, max_reward: float = 1.0):
        self.state_space = template.observation_space
        self.act_space = template.action_space
        self.state_dim = template.observation_dim
        self.act_dim = template.action_dim

        self.reward_func = reward_func
        self.min_reward = min_reward
        self.max_reward = max_reward

        self.uniform_simplex = torch.distributions.Dirichlet(torch.ones(self.state_dim))

        self.randomize()

    def randomize(self):
        self.transition_probabilities = self.uniform_simplex.sample((self.state_dim, self.act_dim))
    
        if self.reward_func is None:
            self.rewards = (torch.rand((self.state_dim, self.act_dim, self.state_dim)) + self.min_reward) * (self.max_reward - self.min_reward)

    @property
    def action_space(self):
        return self.action_space

    @property
    def observation_space(self):
        return self.observation_space

    def sample_initial_state(self):
        raise NotImplementedError

    @extract_one_hot_index_inputs
    def next_state_distribution(self, s, a):
        return self.transition_probabilities[s,a]

    def sample_next_state(self, s, a):
        state_idx = torch.multinomial(self.next_state_distribution(s, a), num_samples=1)
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
        raise NotImplementedError