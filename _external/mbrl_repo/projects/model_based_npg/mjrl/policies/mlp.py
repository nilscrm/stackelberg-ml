import numpy as np
import torch
import torch.nn.functional as F
from mjrl.utils.tensor_utils import tensorize
from torch.autograd import Variable

class MLP(torch.nn.Module):
    def __init__(self, env_spec=None,
                 hidden_sizes=(64,64),
                 min_log_std=-3.0,
                 init_log_std=0.0,
                 seed=123,
                 device='cpu',
                 observation_dim=None,
                 action_dim=None,
                 max_log_std=1.0,
                 *args, **kwargs,
                 ):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        super(MLP, self).__init__()
        # check input specification
        if env_spec is None:
            assert observation_dim is not None
            assert action_dim is not None
        self.observation_dim = env_spec.observation_dim if env_spec is not None else observation_dim   # number of states
        self.action_dim = env_spec.action_dim if env_spec is not None else action_dim                  # number of actions
        self.device = device
        self.seed = seed

        # Set seed
        # ------------------------
        assert type(seed) == int
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Policy network
        # ------------------------
        self.layer_sizes = (self.observation_dim, ) + hidden_sizes + (self.action_dim, )
        self.nonlinearity = torch.tanh
        self.fc_layers = torch.nn.ModuleList([torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
                                             for i in range(len(self.layer_sizes)-1)])
        for param in list(self.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.trainable_params = list(self.parameters())

        # Easy access variables
        # -------------------------
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = torch.zeros(self.observation_dim)

        # Move parameters to device
        # ------------------------
        self.to(device)


    # Network forward
    # ============================================
    def forward(self, observations):
        if type(observations) == np.ndarray: observations = torch.from_numpy(observations).float()
        assert type(observations) == torch.Tensor
        out = observations.to(self.device)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        return F.softmax(out, dim=-1)
    
    def sample_next_action(self, s):
        nads = self.forward(s)

        actions = []
        for i in range(nads.shape[0]):
            nad = nads[i]
            action_idx = torch.multinomial(nad, num_samples=1)
            action = F.one_hot(action_idx, num_classes=self.action_dim)
            actions.append(action)
        res = torch.concat(actions, dim=0)
        return res 


    # Utility functions
    # ============================================
    def to(self, device):
        super().to(device)
        self.trainable_params = list(self.parameters())
        self.device = device

    def get_param_values(self, *args, **kwargs):
        params = torch.cat([p.contiguous().view(-1).data for p in self.parameters()])
        return params.clone()

    def set_param_values(self, new_params: torch.Tensor, *args, **kwargs):
        current_idx = 0
        for idx, param in enumerate(self.parameters()):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            param.data = vals.to(self.device).clone()
            current_idx += self.param_sizes[idx]
        self.trainable_params = list(self.parameters())

    # Main functions
    # ============================================
    def get_action(self, observation):
        assert type(observation) == np.ndarray
        if self.device != 'cpu':
            print("Warning: get_action function should be used only for simulation.")
            print("Requires policy on CPU. Changing policy device to CPU.")
            self.to('cpu')
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        action = self.sample_next_action(self.obs_var).to('cpu').data.numpy().ravel()
        return [action, {'mean': action, 'log_std': 0.0, 'evaluation': action}]

    def mean_LL(self, observations, actions, *args, **kwargs):
        if type(observations) == np.ndarray: observations = torch.from_numpy(observations).float()
        if type(actions) == np.ndarray: actions = torch.from_numpy(actions).float()
        observations, actions = observations.to(self.device), actions.to(self.device)
        action_probs = self.forward(observations)
        action_oh = F.gumbel_softmax(torch.log(actions + 1e-10), hard=True) # NOTE: need to use this instead of sampling to stay differentiable
        LL = -F.nll_loss(torch.log(action_probs + 1e-10), action_oh.argmax(dim=-1), reduction='none') 
        return action_probs, LL

    def mean_kl(self, observations, old_mean):
        new_mean = self.forward(observations)
        return self.kl_divergence(new_mean, tensorize(old_mean))

    def kl_divergence(self, new_mean, old_mean):
        new_mean = new_mean.mean(dim=0, keepdim=True) + 1e-10
        old_mean = old_mean.mean(dim=0, keepdim=True) + 1e-10

        return F.kl_div(new_mean.log(), old_mean)
