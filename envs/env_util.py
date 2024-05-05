import gym

class DiscreteEnv(gym.Env):
    """
        Adds convenience functions to a gym.Env with discrete action and observation space
    """
    @property
    def horizon(self):
        return self.spec.max_episode_steps

    @property
    def observation_dim(self):
        return self.observation_space.n

    @property
    def action_dim(self):
        return self.action_space.n