import gym

from nn.model.model import AWorldModel

class LearnedEnv(gym.Env):
    """ Basically a wrapper, that simulates an environment which is governed by a world model """
    def __init__(self, env_model: AWorldModel, action_space, observation_space):
        self.env_model = env_model

        self.action_space = action_space
        self.observation_space = observation_space

    def reset(self):
        self.state = self.env_model.sample_initial_state()
        return self.state

    def step(self, action):
        reward = self.env_model.reward(self.state, action)
        next_state = self.env_model.sample_next_state(self.state, action)
        done = self.env_model.is_done(next_state)

        self.state = next_state
        return self.state, reward, done, {}

    def render(self):
        print(f"Current state: {self.state}")