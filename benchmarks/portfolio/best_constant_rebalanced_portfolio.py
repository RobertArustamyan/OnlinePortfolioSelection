"""
Best Constant Rebalanced Portfolio (BCRP)
Finds the CRP weights that would have performed best in hindsight.
"""
import numpy as np
from scipy.optimize import minimize


class BestConstantRebalancedPortfolio:
    def __init__(self):
        self.best_weights = None
        self.best_wealth = None
        self.daily_wealth = []
        self.daily_portfolios = []

    def run(self, price_relatives, method='SLSQP', n_restarts=5):
        price_relatives = np.array(price_relatives)
        n_days, n_stocks = price_relatives.shape

        self.best_weights = self._optimize_weights(
            price_relatives,
            method=method,
            n_restarts=n_restarts
        )

        self.daily_wealth = [1.0]
        self.daily_portfolios = [self.best_weights.copy()]

        current_wealth = 1.0
        for day in range(n_days):
            daily_return = np.dot(self.best_weights, price_relatives[day])
            current_wealth *= daily_return
            self.daily_wealth.append(current_wealth)
            self.daily_portfolios.append(self.best_weights.copy())

        self.best_wealth = current_wealth

        return {
            'strategy': 'Best CRP (Hindsight)',
            'final_wealth': self.best_wealth,
            'daily_wealth': np.array(self.daily_wealth),
            'daily_portfolios': np.array(self.daily_portfolios),
            'n_days': n_days,
            'best_weights': self.best_weights,
            'optimization_method': method
        }

    def _optimize_weights(self, price_relatives, method='SLSQP', n_restarts=5):
        n_stocks = price_relatives.shape[1]

        def objective(weights):
            # Take log and negate in order to convert mult to sum and minimize instead of maximizing
            log_wealth = np.sum(np.log(np.dot(weights, price_relatives.T)))
            return -log_wealth

        # Constraints: sum of weights equal 1, all weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_stocks)]

        best_result = None
        best_objective = float('inf')

        # Try multiple random starting points
        for i in range(n_restarts):
            if i == 0:
                # First try: uniform weights
                w0 = np.ones(n_stocks) / n_stocks
            else:
                # Random starting points
                w0 = np.random.dirichlet(np.ones(n_stocks))

            result = minimize(objective, w0, method=method, bounds=bounds, constraints=constraints, options={'maxiter': 1000})

            if result.success and result.fun < best_objective:
                best_objective = result.fun
                best_result = result

        if best_result is None or not best_result.success:
            print("Optimization did not converge well. Using uniform weights")
            return np.ones(n_stocks) / n_stocks

        # Normalize to ensure sum = 1 (numerical precision)
        weights = best_result.x
        weights = weights / np.sum(weights)

        return weights

def run_bcrp(price_relatives, method='SLSQP', n_restarts=5):
    bcrp = BestConstantRebalancedPortfolio()
    return bcrp.run(price_relatives, method=method, n_restarts=n_restarts)