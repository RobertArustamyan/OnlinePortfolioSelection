"""
https://www.schapire.net/papers/newton_portfolios.pdf
"""

import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


class OnlineNewtonStep:
    def __init__(self, n_stocks, eta, beta, delta):
        """
        Online Newton Step algorithm

        :param n_stocks: Number of stocks
        :param eta: Mixing parameter
        :param beta: Learning rate parameter
        :param delta: Regularization parameter
        """
        self.n_stocks = n_stocks
        self.eta = eta
        self.beta = beta
        self.delta = delta

        # Uniform initial portfolio
        self.p_0 = np.ones(n_stocks) / n_stocks

        # Current portfolio state
        self.p_t = self.p_0.copy()

        # Matrices used for cumulative updates
        self.A = np.eye(n_stocks)
        self.b = np.zeros(n_stocks)

    def _project_to_simplex(self, x, M):
        m = M.shape[0]
        P = matrix(2 * M)
        q = matrix(-2 * M @ x)
        G = matrix(-np.eye(m))
        h = matrix(np.zeros((m, 1)))
        A_eq = matrix(np.ones((1, m)))
        b_eq = matrix(1.0)

        sol = solvers.qp(P, q, G, h, A_eq, b_eq)

        if sol['status'] != 'optimal':
            raise RuntimeError(f"CVXOPT optimization failed: {sol['status']}")

        return np.array(sol['x']).flatten()

    def get_portfolio(self):
        """Return mixed portfolio"""
        return (1 - self.eta) * self.p_t + self.eta * self.p_0

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
            self.p_t = self._project_to_simplex(q, self.A)

        except (np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Warning: {type(e).__name__}, using regularization")

            A_reg = self.A + 1e-6 * np.eye(self.n_stocks)

            try:
                A_inv = np.linalg.inv(A_reg)
                q = self.delta * A_inv @ self.b
                q = np.clip(q, -10, 10)
                self.p_t = self._project_to_simplex(q, A_reg)

            except (np.linalg.LinAlgError, RuntimeError) as e2:
                print(f"Warning: Regularization also failed ({type(e2).__name__}), keeping previous portfolio")

    def simulate_trading(self, price_relatives_sequence, verbose=True, verbose_days=100):
        """
        Simulate trading over a sequence of price relatives

        :param price_relatives_sequence:
        :param verbose: Print details (days, wealth, parameters)
        :param verbose_days: number of days to repeat verbose
        """
        wealth = 1.0
        daily_wealth = [1.0]
        portfolios_used = []

        for day, r_t in enumerate(price_relatives_sequence):
            # Get portfolio for current day
            portfolio = self.get_portfolio()
            portfolios_used.append(portfolio.copy())

            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            # Update for next day
            self.update(r_t, portfolio)

            if verbose and ((day + 1) % verbose_days == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}, "
                      f"Portfolio = [{', '.join([f'{x:.3f}' for x in portfolio])}]")

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(portfolios_used),
            'num_days': len(price_relatives_sequence)
        }
