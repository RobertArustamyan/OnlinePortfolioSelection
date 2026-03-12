"""
ons_grid_experiment.py

Grid-search experiment for OnlineNewtonStepCosts.
"""

import numpy as np
from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts
from experiments.portfolio.base_experiment import BaseGridExperiment


class ONSGridExperiment(BaseGridExperiment):
    """
    3-split grid search experiment for the ONS algorithm.
    """

    def __init__(self, cost_model, model_name, stocks,
                 train_data, val_data, test_data,
                 train_prices=None, val_prices=None, test_prices=None,
                 initial_capital=None):
        super().__init__(
            cost_model=cost_model, model_name=model_name, stocks=stocks,
            train_data=train_data, val_data=val_data, test_data=test_data,
            train_prices=train_prices, val_prices=val_prices, test_prices=test_prices,
            initial_capital=initial_capital,
        )
        T = len(train_data) + len(val_data) + len(test_data)
        n = self.n_stocks
        self.eta = (n ** 1.25) / (np.sqrt(T * np.log(n * T)))
        self.beta = 1.0 / (8 * (n ** 0.25) * np.sqrt(T * np.log(n * T)))

    @property
    def initial_portfolio_key(self) -> str:
        return 'ons_initial_portfolio'

    def _build_algorithm(self, params: dict, cost_model) -> OnlineNewtonStepCosts:
        return OnlineNewtonStepCosts(
            n_stocks=self.n_stocks,
            eta=self.eta,
            beta=self.beta,
            delta=params['delta'],
            cost_model=cost_model,
            cost_penalty=params['cost_penalty'],
            alpha=params['alpha'],
        )

    def _transfer_state(self, algo_src: OnlineNewtonStepCosts,
                        algo_dst: OnlineNewtonStepCosts):
        algo_dst.A = algo_src.A.copy()
        algo_dst.b = algo_src.b.copy()
        algo_dst.p_t = algo_src.p_t.copy()
        algo_dst.prev_portfolio = algo_src.prev_portfolio.copy()

    def _param_combinations(self, parameter_grid: dict) -> list:
        return [
            {'delta': d, 'cost_penalty': cp, 'alpha': a}
            for d in parameter_grid['deltas']
            for cp in parameter_grid['cost_penalties']
            for a in parameter_grid['alphas']
        ]