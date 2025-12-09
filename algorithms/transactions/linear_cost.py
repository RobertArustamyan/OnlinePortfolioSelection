from typing import List

import numpy as np

from algorithms.transactions.cost import Costs


class LinearCost(Costs):
    def __init__(self, gamma: float):
        self.gamma = gamma

    def compute_cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        turnover = np.sum(np.abs(new_weights - prev_weights))
        cost = self.gamma * turnover

        return cost

    def gradient(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
        diff = new_weights - prev_weights
        sign = np.sign(diff)
        sign[np.isclose(diff, 0.0, atol=1e-12)] = 0.0

        return self.gamma * sign