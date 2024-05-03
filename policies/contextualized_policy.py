import numpy as np

from nn.model.world_models import AWorldModel
from policies.policy import APolicy

class ContextualizedPolicy(APolicy):
    """ Basically a wrapper that appends a context to the observation when choosing an action """

    def __init__(self, policy: APolicy, initial_context: np.ndarray):
        self.policy = policy
        self.context = initial_context

    def set_context(self, context: np.ndarray):
        self.context = context
    
    def sample_next_action(self, observation: np.ndarray) -> np.ndarray:
        return self.policy.sample_next_action(np.concatenate([observation, self.context], axis=0))
    
class ModelContextualizedPolicy(ContextualizedPolicy):
    """ Contextualized policy that uses queries to a world model as context """

    def __init__(self, policy: APolicy, dynamics_queries = [], reward_queries = []):
        super().__init__(policy, None)
        self.dynamics_queries = dynamics_queries
        self.reward_queries = reward_queries

        self.set_context_by_querying()

    def set_context_by_querying(self, env_model: AWorldModel):
        query_answers = []
        
        for (s, a) in self.dynamics_queries:
            query_answers.append(env_model.next_state_distribution(s, a))
        for (s, a, s_next) in self.reward_queries:
            query_answers.append(env_model.reward(s, a, s_next))

        self.set_context(np.array(query_answers).reshape(-1))