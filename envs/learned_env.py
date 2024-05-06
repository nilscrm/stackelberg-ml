from envs.env_util import AEnv
from nn.model.world_models import AWorldModel

from util.tensor_util import one_hot_to_idx, one_hot

# NOTE: Discrete envs work on integers in our case
class DiscreteLearnedEnv(AEnv):
    """ Basically a wrapper, that simulates an environment which is governed by a world model """
    def __init__(self, env_model: AWorldModel, action_space, observation_space, max_episode_steps=100000):
        self.env_model = env_model

        self._action_dim = env_model.action_dim
        self._observation_dim = env_model.observation_dim
        self._max_episode_steps = max_episode_steps

        self.action_space = action_space
        self.observation_space = observation_space

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def action_dim(self):
        return self._action_dim

    def reset(self, seed=None) -> int:
        # TODO: could query here if we wanted it to happen automatically (e.g. when training with trainers from other libraries)
        self.state = one_hot_to_idx(self.env_model.sample_initial_state())
        return self.state, {}

    def step(self, action: int):
        old_state = one_hot(self.state, self.observation_dim)

        self.state = one_hot_to_idx(self.env_model.sample_next_state(old_state, action))
        return self.state, self.env_model.reward(old_state, action, self.state), self.env_model.is_done(self.state), False, {} # TODO: do correct truncation

    def render(self):
        print(f"Current state: {self.state}")