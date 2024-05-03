from abc import ABC, abstractmethod

class APolicy(ABC):
    
    @abstractmethod
    def next_action_distribution(self, observation):
        pass

    @abstractmethod
    def sample_next_action(self, observation):
        pass