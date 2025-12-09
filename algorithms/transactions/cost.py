from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Costs(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_cost(self, prev_weights, new_weights) -> float:
        raise NotImplementedError

    @abstractmethod
    def gradient(self, prev_weights, new_weights) -> List[float]:
        raise NotImplementedError

    def numerical_gradient(self, prev_weights, new_weights, eps=1e-6) -> List[float]:
        g = np.zeros_like(new_weights, dtype=float)
        for i in range(len(new_weights)):
            p_plus = new_weights.copy()
            p_minus = new_weights.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            c_plus = self.compute_cost(prev_weights, p_plus)
            c_minus = self.compute_cost(prev_weights, p_minus)
            g[i] = (c_plus - c_minus) / (2 * eps)
        return g