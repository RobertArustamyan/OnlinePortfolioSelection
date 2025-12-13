import numpy as np
import cvxpy as cp

from algorithms.transactions.cost import Costs


class QuadraticCost(Costs):
    def __init__(self, lambda_param: float):
        self.lambda_param = lambda_param

    def compute_cost(self, prev_weights, new_weights):
        return self.lambda_param * np.sum((new_weights - prev_weights) ** 2)

    def compute_gradient(self, prev_weights, new_weights):
        return self.lambda_param * cp.sum((new_weights - prev_weights) ** 2)

    def gradient(self, prev_weights, new_weights):
        return 2.0 * self.lambda_param * (new_weights - prev_weights)