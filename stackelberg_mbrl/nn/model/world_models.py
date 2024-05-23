from typing import Callable, Optional, Literal

from abc import abstractmethod
import gymnasium
import numpy as np
from pathlib import Path
import torch
from torch.nn import functional as F

from stackelberg_mbrl.envs.env_util import draw_mdp
from stackelberg_mbrl.nn.mlp import MLP
from stackelberg_mbrl.nn.model.dynamics_nets import DynamicsNetMLP
from stackelberg_mbrl.nn.model.reward_nets import RewardNetMLP

from stackelberg_mbrl.util.optimization import fit, fit_tuple
from stackelberg_mbrl.util.tensor_util import extract_one_hot_index_inputs, tensorize, tensorize_array_inputs, one_hot

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

    def query(self, dynamics_queries = [], reward_queries = []):
        """ Queries this model and gets all the answers as a flat tensor """
        query_answers = []        

        for (s, a) in dynamics_queries:
            query_answers.append(self.next_state_distribution(s, a))

        for (s, a, s_next) in reward_queries:
            query_answers.append(torch.tensor([self.reward(s, a, s_next)]))

        return torch.concatenate(query_answers).flatten()

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
            return self.reward_function(s, a, s_next)
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
    
    def draw_mdp(self, filepath: Path, format: Literal['png', 'svg'] = 'png'):
        with torch.no_grad():
            transition_probs = []
            for s in range(self.observation_dim):
                transition_probs.append([])
                for a in range(self.action_dim):
                    action = one_hot(a, num_classes=self.act_dim).float()
                    state = one_hot(s, num_classes=self.observation_dim).float()
                    transition_probs[-1].append(self.next_state_distribution(state, action).numpy())
                        
            rewards = [[[self.reward(s, a, s_next) for s_next in range(self.observation_dim)] for a in range(self.action_dim)] for s in range(self.observation_dim)]

        draw_mdp(np.array(transition_probs), np.array(rewards), filepath, format)

class StaticDiscreteModel(AWorldModel):
    """ Random model of the world with a discrete action and observation space that has non-learnable transition probabilities """
    def __init__(self, template, init_state_probs: np.ndarray, 
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

    def set_transition_probs(self, transition_probabilities: torch.Tensor):
        self.transition_probabilities = transition_probabilities

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
        return self.transition_probabilities[s][a]

    def sample_next_state(self, s, a):
        state_idx = torch.multinomial(self.next_state_distribution(s, a), num_samples=1)[0] # convert to non-batched
        return F.one_hot(state_idx, num_classes=self.state_dim)

    def reward(self, s, a, s_next):
        if self.reward_func is None:
            return self._static_rewards(s, a, s_next)
        else:
            return self.reward_func(s,a,s_next)
    
    @extract_one_hot_index_inputs
    def _static_rewards(self, s, a, s_next):
        return self.rewards[s,a,s_next]

    def is_done(self, s):
        return self.termination_func(s)
    
    def draw_mdp(self, filepath: Path, format: Literal['png', 'svg'] = 'png'):
        rewards = np.array([[[self.reward(s, a, s_next) for s_next in range(self.observation_dim)] for a in range(self.action_dim)] for s in range(self.observation_dim)])
        transition_probabilities = np.array([[self.transition_probabilities[s][a].numpy() for a in range(self.action_dim)] for s in range(self.observation_dim)])
        draw_mdp(transition_probabilities, rewards, filepath, format)



class ContextualizedWorldModel(AWorldModel):
    """ 
        Model that operates on contextualized states

        - Dynamics: (context, s), a -> (context, s_next) [actually only predicts s_next and simply copies the context forward]
        - Reward: (context, s), a -> r
    """
    def __init__(self, context_size, state_dim, act_dim,
                 hidden_sizes=(64,64),
                 activation=torch.relu,
                 rewards: np.ndarray=None, # reward will be learned if no rewards are provided
                 fit_lr=1e-3,
                 fit_weight_decay=0.0):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_size = context_size

        # dynamics
        self.dynamics_net = MLP(
            input_dim=context_size + state_dim + act_dim,
            hidden_sizes=hidden_sizes,
            output_dim=state_dim,
            nonlinearity=activation
        )
        self.dynamics_opt = torch.optim.Adam(self.dynamics_net.parameters(), lr=fit_lr, weight_decay=fit_weight_decay)
        self.dynamics_loss = torch.nn.CrossEntropyLoss()

        # rewards
        self.rewards = rewards
        if rewards is None:
            self.rewards_net = MLP(
                input_dim=context_size + state_dim + act_dim + state_dim,
                hidden_sizes=hidden_sizes,
                output_dim=1,
                nonlinearity=activation
            )
            self.rewards_opt = torch.optim.Adam(self.rewards_net.parameters(), lr=fit_lr, weight_decay=fit_weight_decay)
            self.rewards_loss = torch.nn.MSELoss()
        
    @property
    def action_dim(self):
        return self.act_dim

    @property
    def observation_dim(self):
        return self.state_dim

    def _split_context_state(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            return observation[:self.context_size], observation[self.context_size:]
        elif observation.ndim == 2:
            return observation[:,:self.context_size], observation[:,self.context_size:]


    @tensorize_array_inputs
    def next_state_distribution(self, observation, action):
        # NOTE: really only returns the distribution over next states (no context!)
        x = torch.cat([observation, action])
        out = self.dynamics_net.forward(x)
        return F.softmax(out, dim=-1)
    
    @tensorize_array_inputs
    def sample_next_state(self, observation, action):
        context,_ = self._split_context_state(observation)
        state_idx = torch.multinomial(self.next_state_distribution(observation, action), num_samples=1)
        out = F.one_hot(state_idx, num_classes=self.state_dim)[0] # convert to non-batched
        return torch.cat([context, out])

    @tensorize_array_inputs
    def reward(self, observation, action, observation_next):
        context, state = self._split_context_state(observation)
        _,state_next = self._split_context_state(observation_next)

        if self.rewards is not None:
            state_idx = state.argmax().item()
            action_idx = action.argmax().item()
            state_next_idx = state_next.argmax().item()
            return self.rewards[state_idx, action_idx, state_next_idx]
        else:
            x = torch.cat([context, state, action, state_next])
            return self.rewards_net.forward(x)

    @tensorize_array_inputs
    def fit_dynamics(self, observations, actions, observations_next, fit_mb_size, fit_epochs, max_steps=1e4):
        # TODO: also try kldiv like they do in the paper (bc CE will never be 0, as we sample)
        _,states_next = self._split_context_state(observations_next)
        X = torch.cat([observations, actions], dim=-1)
        Y = states_next.argmax(dim=-1)
        return fit(self.dynamics_net, X, Y, self.dynamics_opt, self.dynamics_loss, 
                         fit_mb_size, fit_epochs, max_steps=max_steps)

    @tensorize_array_inputs
    def fit_reward(self, observations, actions, observations_next, rewards, fit_mb_size, fit_epochs, max_steps=1e4):
        assert self.rewards is None, "Reward model was not initialized to be learnable. Using the rewards from the env."
        contexts, states = self._split_context_state(observations)
        _,states_next = self._split_context_state(observations_next)

        X = torch.cat([contexts, states, actions, states_next], dim=-1)
        Y = rewards
        return fit(self.rewards_net, X, Y, self.rewards_opt, self.rewards_loss,
                         fit_mb_size, fit_epochs, max_steps=max_steps)