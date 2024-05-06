from policies.policy import APolicy
from stable_baselines3.common.policies import BasePolicy
from util.tensor_util import extract_one_hot_index_inputs, one_hot

class SB3DiscretePolicy(APolicy):
    """ Wraps a SB3 discrete policy as an APolicy (so it can be used with the rest of our code) """
    def __init__(self, policy: BasePolicy) -> None:
        super().__init__()

        self.policy = policy


    def next_action_distribution(self, observation):
        raise NotImplementedError

    @extract_one_hot_index_inputs
    def sample_next_action(self, observation: int):
        action, _ = self.policy.predict(observation)
        return one_hot(int(action), int(self.policy.observation_space.n)) # need to convert to int bc one_hot does not support int64




    