import numpy as np

from nn.model.world_models import AWorldModel
from policies.policy import APolicy

class ContextualizedPolicy(APolicy):
    """ Basically a wrapper that appends a context to the observation when choosing an action """

    def __init__(self, policy: APolicy, initial_context: np.ndarray):
        super().__init__()
        self.policy = policy
        self.context = initial_context

    def set_context(self, context: np.ndarray):
        self.context = context

    @property
    def trainable_params(self):
        return self.policy.trainable_params
    
    def next_action_distribution(self, observation):
        return self.policy.next_action_distribution(np.concatenate([observation, self.context], axis=0))

    def sample_next_action(self, observation):
        return self.policy.sample_next_action(np.concatenate([observation, self.context], axis=0))
    
    def log_likelihood(self, observations, groundtruth_actions):
        return self.policy.log_likelihood(self._append_context(observations), groundtruth_actions)

    def kl_divergence(self, observations, old_actions):
        return self.policy.kl_divergence(self._append_context(observations), old_actions)
    
    def _append_context(self, observations):
        rows,cols = observations.shape
        tiled_vector = np.tile(self.context, (rows, 1))
        observation_with_context = np.zeros((rows, cols + len(self.context)))
        observation_with_context[:,:cols] = observations
        observation_with_context[:,cols:] = tiled_vector
        return observation_with_context
        
class ModelContextualizedPolicy(ContextualizedPolicy):
    """ Contextualized policy that uses queries to a world model as context """

    def __init__(self, policy: APolicy, dynamics_queries = [], reward_queries = []):
        super().__init__(policy, None)
        self.dynamics_queries = dynamics_queries
        self.reward_queries = reward_queries

        self.context = None # NOTE: make sure to call set_context_by_querying before sampling from the contextualized policy

    def set_context_by_querying(self, env_model: AWorldModel):
        query_answers = []        
        for (s, a) in self.dynamics_queries:
            query_answers.extend(env_model.next_state_distribution(s, a))
        for (s, a, s_next) in self.reward_queries:
            query_answers.append(env_model.reward(s, a, s_next))

        self.set_context(np.array(query_answers).ravel())