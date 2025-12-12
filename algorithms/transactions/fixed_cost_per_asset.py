import numpy as np

from algorithms.transactions.cost import Costs


class FixedCostPerAsset(Costs):
    def __init__(self, cost_per_transaction: float):
        self.cost_per_transaction = cost_per_transaction

    def compute_cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        n_trades = np.sum(np.abs(prev_weights - new_weights) >= 1e-12)
        return self.cost_per_transaction * n_trades

    def gradient(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
        diff = new_weights - prev_weights
        sign = np.sign(diff)
        sign[np.isclose(diff, 0.0, atol=1e-12)] = 0.0

        return self.cost_per_transaction * sign

class FixedCostPerAssetSmooth(Costs):
    def __init__(self, cost_per_transaction: float, smoothness: float = 100.0):
        self.cost_per_transaction = cost_per_transaction
        self.smoothness = smoothness

    def compute_cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        diff = np.abs(new_weights - prev_weights)
        smooth_indicator = np.tanh(self.smoothness * diff)

        return self.cost_per_transaction * np.sum(smooth_indicator)

    def gradient(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> np.ndarray:
        diff = new_weights - prev_weights
        abs_diff = np.abs(diff)

        k = self.smoothness
        sech_squared = 1.0 / np.cosh(k * abs_diff) ** 2
        sign = np.sign(diff)

        return self.cost_per_transaction * k * sech_squared * sign