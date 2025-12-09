import numpy as np

from algorithms.transactions.cost import Costs


class LinearCost(Costs):
    def __init__(self, gamma: float):
        self.gamma = gamma

    def compute_cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        turnover = np.sum(np.abs(new_weights - prev_weights))
        cost = self.gamma * turnover
        return cost
