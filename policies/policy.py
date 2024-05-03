from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np

class APolicy(ABC):
    
    @abstractmethod
    def get_action(self, observation) -> Tuple[np.ndarray, Dict]:
        pass