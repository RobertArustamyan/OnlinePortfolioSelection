from abc import ABC, abstractmethod

class Costs(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_cost(self, prev_weights, new_weights) -> float:
        pass
