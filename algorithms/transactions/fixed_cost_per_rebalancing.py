import numpy as np

from algorithms.transactions.cost import Costs


class FixedCostPerRebalancing(Costs):
    def __init__(self, cost_per_transaction: float):
        self.cost_per_transaction = cost_per_transaction

    def compute_cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        if np.any(np.abs(new_weights - prev_weights) > 1e-12):
            return self.cost_per_transaction
        return 0.0

    def gradient(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        pass


class FixedCostPerRebalancing(Costs):
    def __init__(self, cost_per_transaction: float):
        self.cost_per_transaction = cost_per_transaction

    def compute_cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        if np.any(np.abs(new_weights - prev_weights) > 1e-12):
            return self.cost_per_transaction
        return 0.0

    def gradient(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
        return np.zeros_like(new_weights)


class FixedCostPerRebalancingSmooth(Costs):
    def __init__(self, cost_per_transaction: float, smoothness: float = 50.0):
        self.cost_per_transaction = cost_per_transaction
        self.smoothness = smoothness

    def compute_cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        turnover = np.sum(np.abs(new_weights - prev_weights))
        # tanh smoothly goes from 0 to 1
        return self.cost_per_transaction * np.tanh(self.smoothness * turnover)

    def gradient(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
        diff = new_weights - prev_weights
        turnover = np.sum(np.abs(diff))

        k = self.smoothness
        c = self.cost_per_transaction

        # Gradient: c * k * sech²(k*turnover) * sign(diff)
        sech_squared = 1.0 / np.cosh(k * turnover) ** 2
        sign = np.sign(diff)

        return c * k * sech_squared * sign