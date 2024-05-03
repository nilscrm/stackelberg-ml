import numpy as np

from models.nn_dynamics import AWorldModel
from policies.policy import APolicy

class ContextualizedPolicy(APolicy):
    """ Basically a wrapper that appends a context to the observation when choosing an action """

    def __init__(self, policy: APolicy, initial_context: np.ndarray):
        self.policy = policy
        self.context = initial_context

    def set_context(self, context: np.ndarray):
        self.context = context
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        return super().get_action(np.concatenate([observation, self.context], axis=0))
    
class ModelContextualizedPolicy(ContextualizedPolicy):
    """ Contextualized policy that uses queries to a model as context """

    def __init__(self, policy: APolicy, queries):
        super().__init__(policy, None)
        self.queries = queries

        self.set_context_by_querying()

    def set_context_by_querying(self, env_model: AWorldModel):
        # Query model
        query_answers = []
        for (s, a) in self.queries:
            query_answers.append(env_model.next_state_distribution(s, a))

        self.set_context(np.array(query_answers).reshape(-1))