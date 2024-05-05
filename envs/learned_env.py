import gym

from nn.model.world_models import AWorldModel

class LearnedEnv(gym.Env):
    """ Basically a wrapper, that simulates an environment which is governed by a world model """
    def __init__(self, env_model: AWorldModel, action_space, observation_space):
        self.env_model = env_model

        self.action_space = action_space
        self.observation_space = observation_space

    def reset(self):
        # TODO: could query here if we wanted it to happen automatically (e.g. when training with trainers from other libraries)
        self.state = self.env_model.sample_initial_state().numpy()
        return self.state

    def step(self, action):
        old_state = self.state
        self.state = self.env_model.sample_next_state(self.state, action)
        return self.state.numpy(), self.env_model.reward(old_state, action, self.state).numpy(), self.env_model.is_done(self.state), {}

    def render(self):
        print(f"Current state: {self.state}")