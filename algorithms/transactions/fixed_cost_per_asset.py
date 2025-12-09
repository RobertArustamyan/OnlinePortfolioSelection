import numpy as np

from algorithms.transactions.cost import Costs


class FixedCostPerAsset(Costs):
    def __init__(self, cost_per_transaction: float):
        self.cost_per_transaction = cost_per_transaction

    def compute_cost(self, prev_weights: np.ndarray, new_weights: np.ndarray) -> float:
        n_trades = np.sum(np.abs(prev_weights - new_weights) >= 1e-12)

        return self.cost_per_transaction * n_trades