from abc import ABC
import gymnasium

class AEnv(ABC, gymnasium.Env):
    def __init__(self) -> None:
        gymnasium.Env.__init__(self)

    @property
    def max_episode_steps(self):
        pass

    @property
    def observation_dim(self):
        pass

    @property
    def action_dim(self):
        pass


class DiscreteEnv(AEnv):
    """
        Adds convenience functions to a gym.Env with discrete action and observation space
    """
    def __init__(self, max_episode_steps) -> None:
        super().__init__()

        self._max_episode_steps = max_episode_steps

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def observation_dim(self):
        return self.observation_space.n

    @property
    def action_dim(self):
        return self.action_space.n