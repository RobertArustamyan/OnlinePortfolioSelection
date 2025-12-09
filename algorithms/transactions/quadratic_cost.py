import numpy as np

from algorithms.transactions.cost import Costs


class QuadraticCost(Costs):
    def __init__(self, lambda_param: float):
        self.lambda_param = lambda_param

    def compute_cost(self, prev_weights, new_weights):
        return self.lambda_param * np.sum((new_weights - prev_weights) ** 2)