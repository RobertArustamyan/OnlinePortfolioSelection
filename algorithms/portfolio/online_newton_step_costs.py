"""
https://www.schapire.net/papers/newton_portfolios.pdf
"""

import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers
import cvxpy as cp

solvers.options['show_progress'] = False


class OnlineNewtonStepCosts:
    def __init__(self, n_stocks, eta, beta, delta, cost_model, cost_penalty=1.0, alpha=0.0):
        """
        Cost-aware Online Newton Step algorithm.
        :param n_stocks: Number of stocks
        :param eta: Mixing parameter
        :param beta: Learning rate parameter
        :param delta: Regularization parameter
        :param cost_model: Transaction cost model
        :param cost_penalty: Weight for transaction cost in optimization
                         Higher = more conservative trading
        :param alpha: No-trade threshold. Only rebalance if portfolio change exceeds alpha.
                   Measured as L1 distance: sum(|new_weights - old_weights|)
        """
        self.n_stocks = n_stocks
        self.eta = eta
        self.beta = beta
        self.delta = delta
        self.cost_model = cost_model
        self.cost_penalty = cost_penalty
        self.alpha = alpha

        # Uniform initial portfolio
        self.p_0 = np.ones(n_stocks) / n_stocks

        # Current portfolio state
        self.p_t = self.p_0.copy()
        self.prev_portfolio = self.p_0.copy()

        # Matrices used for cumulative updates
        self.A = np.eye(n_stocks)
        self.b = np.zeros(n_stocks)

        self.A_history = []
        self.b_history = []
        self.mahalanobis_history = []
        self.objective_history = []

    def _compute_objective_value(self, portfolio, target, M, prev_portfolio):
        # Mahalanobis distance term: ||p - x||^2_M
        deviation = portfolio - target
        mahalanobis = deviation @ M @ deviation
        # Transaction cost term: cost(prev, p)
        transaction_cost = self.cost_model.compute_cost(prev_portfolio, portfolio)

        return mahalanobis + self.cost_penalty * transaction_cost

    def _project_to_simplex_standard(self, x, M):
        """

        :param x:
        :param M:
        """
        n = M.shape[0]

        # Decision variable
        p = cp.Variable(n)

        objective = cp.Minimize(
            0.5 * cp.quad_form(p, 2 * M) + (-2 * M @ x) @ p
        )

        constraints = [
            p >= 0,
            cp.sum(p) == 1
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"CVXPY optimization failed: {prob.status}")

        return p.value

    def _cost_aware_objective_factory(self, x, M, prev_portfolio):
        """
        Function that creates the cost-aware objective for existing cost model.
        returns a function: f(p) = ||p - x||^2_M + lambda * cost(prev, p)
        """

        def objective(p):
            # Mahalanobis distance term: ||p - x||^2_M
            deviation = p - x
            mahalanobis_distance = deviation @ M @ deviation

            # Transaction cost term: cost(prev, p)
            transaction_cost = self.cost_model.compute_cost(prev_portfolio, p)

            # Combined objective
            # THIS PART IS TESTING!
            scale = np.trace(M)
            #
            return (1 / scale) * mahalanobis_distance + self.cost_penalty * transaction_cost

        return objective

    def _cost_aware_gradient_factory(self, x, M, prev_portfolio):
        """
        Factory function for gradient of cost-aware objective.
        Uses numerical differentiation for cost gradient.
        """
        scale = np.trace(M)
        def gradient(p):
            mahalanobis_grad = (2 / scale) * M @ (p - x)

            try:
                cost_grad = self.cost_model.gradient(prev_portfolio, p)
            except (NotImplementedError, AttributeError):
                cost_grad = self.cost_model.numerical_gradient(prev_portfolio, p)

            return mahalanobis_grad + self.cost_penalty * cost_grad

        return gradient

    def _project_to_simplex_with_costs(self, x, M, prev_portfolio):
        """
        Cost-aware projection
        """
        # create objective and gradient functions
        objective = self._cost_aware_objective_factory(x, M, prev_portfolio)
        gradient = self._cost_aware_gradient_factory(x, M, prev_portfolio)

        constraints = [
            {'type': 'eq', 'fun': lambda p: np.sum(p) - 1},
        ]
        bounds = [(0, 1) for _ in range(self.n_stocks)]

        # Initial guess
        p0 = self._project_to_simplex_standard(x, M)
        result = minimize(
            objective,
            p0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6, 'maxiter': 10000}
        )

        if result.success:
            return result.x
        else:
            return self._project_to_simplex_standard(x, M)

    def get_portfolio(self):
        """
        Return mixed portfolio for trading.
        Applies no-trade threshold if alpha > 0.
        """
        # ONS suggested portfolio
        target_portfolio = self.delta * np.linalg.inv(self.A) @ self.b
        proposed_portfolio = (1 - self.eta) * self.p_t + self.eta * self.p_0

        # No-trade threshold
        if self.alpha > 0:
            portfolio_change = np.sum(np.abs(proposed_portfolio - self.prev_portfolio))

            if portfolio_change < self.alpha:
                return self.prev_portfolio.copy()

        return proposed_portfolio

    def update(self, r_t, p_used):
        """
        Update algorithm state after observing price relatives r_t

        :param r_t: price relative
        :param p_used: previous portfolio (already used)
        """
        # Compute gradient
        wealth = np.dot(p_used, r_t)

        if wealth <= 0:
            print(f"Warning: Non-positive wealth {wealth}, skipping update")
            return

        grad = r_t / wealth

        # Update A and b
        self.A += np.outer(grad, grad)
        self.b += (1 + 1.0 / self.beta) * grad

        try:
            A_inv = np.linalg.inv(self.A)
            q = self.delta * A_inv @ self.b
            self.p_t = self._project_to_simplex_with_costs(q, self.A, self.prev_portfolio)

        except (np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Warning: {type(e).__name__}")

            A_reg = self.A + 1e-6 * np.eye(self.n_stocks)

            try:
                A_inv = np.linalg.inv(A_reg)
                q = self.delta * A_inv @ self.b
                q = np.clip(q, -10, 10)
                self.p_t = self._project_to_simplex_with_costs(q, A_reg, self.prev_portfolio)

            except (np.linalg.LinAlgError, RuntimeError) as e2:
                print(f"Warning: Regularization failed ({type(e2).__name__}) keeping previous portfolio")

    def simulate_trading(self, price_relatives_sequence, stock_prices_sequence=None, verbose=True, verbose_days=100):
        """
        Simulate trading over a sequence of price relatives

        :param price_relatives_sequence:price relatives of stocks used for trading
        :param stock_prices_sequence: prices of stocks used for trading
        :param verbose: Print details (days, wealth, parameters)
        :param verbose_days: number of days to repeat verbose
        """
        wealth = 1.0
        daily_wealth = [1.0]
        portfolios_used = []
        transaction_costs = []
        turnovers = []
        num_no_trades = 0
        num_alpha_blocks = 0
        num_improvement_blocks = 0
        block_reasons = []

        for day, r_t in enumerate(price_relatives_sequence):
            if hasattr(self.cost_model, 'update_state'):
                if stock_prices_sequence is not None:
                    day_prices = stock_prices_sequence[day]
                else:
                    raise ValueError("Something went wrong in hasattr if statement")
                self.cost_model.update_state(wealth_fraction=wealth, stock_prices=day_prices)

            # Get portfolio for current day (with thresholds)
            portfolio = self.get_portfolio()

            # Metrics used for visualisation only!
            self.A_history.append(self.A.copy())
            self.b_history.append(self.b.copy())
            target_portfolio = self.delta * np.linalg.inv(self.A) @ self.b
            deviation = portfolio - target_portfolio
            mahalanobis = deviation @ self.A @ deviation
            self.mahalanobis_history.append(mahalanobis)


            actually_traded = not np.allclose(portfolio, self.prev_portfolio)
            if not actually_traded:
                num_no_trades += 1
                # Try to determine which threshold blocked
                portfolio_change = np.sum(np.abs(self.p_t - self.prev_portfolio))
                if self.alpha > 0 and portfolio_change < self.alpha:
                    num_alpha_blocks += 1
                    block_reasons.append('alpha')
                else:
                    block_reasons.append('other')
            else:
                block_reasons.append('traded')

            # Compute transaction cost for the rebalancing
            tc = self.cost_model.compute_cost(self.prev_portfolio, portfolio)

            # Apply transaction cost to wealth
            wealth *= (1 - tc)
            transaction_costs.append(tc)

            turnover = np.sum(np.abs(portfolio - self.prev_portfolio))
            turnovers.append(turnover)

            portfolios_used.append(portfolio.copy())

            # Apply market returns
            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            # Update algorithm state for next day
            self.update(r_t, portfolio)
            self.prev_portfolio = portfolio.copy()

            if verbose and ((day + 1) % verbose_days == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}, "
                      f"TC = {transaction_costs[-1]:.6f}, "
                      f"Turnover = {turnover:.4f}, "
                      f"No-trades: {num_no_trades} (alpha:{num_alpha_blocks}, imp:{num_improvement_blocks})")

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(portfolios_used),
            'transaction_costs': np.array(transaction_costs),
            'turnovers': np.array(turnovers),
            'total_transaction_cost': sum(transaction_costs),
            'avg_turnover': np.mean(turnovers),
            # 'max_turnover': np.max(turnovers),
            'max_turnover': float(np.max(turnovers)) if len(turnovers) > 0 else 0.0,
            'num_days': len(price_relatives_sequence),
            'num_no_trades': num_no_trades,
            'num_alpha_blocks': num_alpha_blocks,
            'num_improvement_blocks': num_improvement_blocks,
            'trade_frequency': 1 - (num_no_trades / len(price_relatives_sequence)),
            'block_reasons': block_reasons,
            'A_history': self.A_history,
            'b_history': self.b_history,
            'mahalanobis_history': self.mahalanobis_history
        }
