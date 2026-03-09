"""
https://arxiv.org/pdf/1805.07430
"""

import numpy as np
import cvxpy as cp

from algorithms.portfolio.barrons import Barrons

class AdaBarrons:
    def __init__(self, n_stocks, T):
        """
        Ada-Barrons algorithm.

        :param n_stocks: Number of stocks
        :param T: Time horizon
        """
        self.n_stocks = n_stocks
        self.T = T
        self.eta = 1.0 / (2048 * n_stocks * (np.log(T)**2))
        self.beta = 0.5
        self.gamma = 0.04 # 1 / 25

        self.barrons = Barrons(n_stocks, T, self.beta, self.eta)

        self.global_portfolios_used = []
        self.global_observed_loss = []
        self.restart_count = 0

        self.cumulative_losses = []

        self.observed_price_relatives = []
        self.observed_portfolios = []

    def _compute_u_t(self):
        """
        Compute the hindsight-optimal portfolio u_t.
        :return: Optimal portfolio u_t as a numpy array
        """
        if len(self.observed_price_relatives) == 0:
            print("entered comput_u_t function without loss")
            return np.ones(self.n_stocks) / self.n_stocks

        u = cp.Variable(self.n_stocks)

        R = np.array(self.observed_price_relatives)
        loss_term = -cp.sum(cp.log(R @ u))

        reg_term = (1.0 / self.gamma) * cp.sum(-cp.log(u))
        objective = cp.Minimize(loss_term + reg_term)

        constraints = [cp.sum(u) == 1, u >= self.barrons.x_min]
        prob = cp.Problem(objective, constraints)

        # Using different solvers as, one single solver may fail
        try:
            prob.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-6, feastol=1e-8, verbose=False)

            if u.value is None:
                raise RuntimeError("ECOS failed to find solution")

            u_val = np.maximum(u.value, self.barrons.x_min)
            u_val /= u_val.sum()
            return u_val
        except:
            try:
                prob.solve(solver=cp.CVXOPT, verbose=False)

                if u.value is None:
                    raise RuntimeError("CVXOPT failed to find solution")

                u_val = np.maximum(u.value, self.barrons.x_min)
                u_val /= u_val.sum()
                return u_val
            except:
                try:
                    prob.solve(solver=cp.SCS, eps=1e-5, max_iters=5000, verbose=False)

                    if u.value is None:
                        print("Warning: All solvers (ECOS, CVXOPT, SCS) failed. Using uniform portfolio.")
                        return np.ones(self.n_stocks) / self.n_stocks

                    u_val = np.maximum(u.value, self.barrons.x_min)
                    u_val /= u_val.sum()
                    return u_val
                except:
                    print("Warning: All solvers failed with exceptions. Using uniform portfolio.")
                    return np.ones(self.n_stocks) / self.n_stocks

    def _compute_alpha_t(self, u_t):
        """
        Compute the adaptive step-size upper bound alpha_t. Is used to decide whether the current beta value
        is still valid or a restart is needed.

        :param u_t: Hindsight-optimal portfolio computed
        """
        if len(self.observed_price_relatives) == 0:
            print("Entered compute_alpha_t function without loss")
            return 0.5

        min_alpha = 0.5

        for s in range(len(self.observed_price_relatives)):
            r_s = self.observed_price_relatives[s]
            x_s = self.observed_portfolios[s]

            # Compute delta_s = gradient of f_s at x_s = -r_s / <x_s, r_s>
            delta_s = -r_s / np.dot(x_s, r_s)

            # Compute |(u_t - x_s)^T delta_s|
            diff = u_t - x_s
            inner_product = np.dot(diff, delta_s)
            abs_inner_product = np.abs(inner_product)

            # Avoid division by zero
            if abs_inner_product > 1e-10:
                alpha_s = 1.0 / (8.0 * abs_inner_product)
                min_alpha = min(min_alpha, alpha_s)

        return min_alpha

    def get_portfolio(self):
        """Return portfolio for trading"""
        return self.barrons.get_portfolio()

    def _check_restart_condition(self):
        """
        Checks if the algorithm restart condition is satisfied (if yes, makes beta = beta / 2 and restarts)
        :return: true if condition satisfied false otherwise
        """
        if len(self.observed_price_relatives) == 0:
            print("entered restart condition without loss")
            return False

        u_t = self._compute_u_t()
        alpha_t = self._compute_alpha_t(u_t)

        return self.beta > alpha_t

    def _restart_barrons(self):
        """
        Updates parameters and resets algorithm with new parameters, when the condition satisfied
        """
        self.beta = self.beta / 2.0
        self.barrons = Barrons(n_stocks=self.n_stocks, T=self.T, beta=self.beta, eta=self.eta)

        self.observed_price_relatives = []
        self.observed_portfolios = []
        self.restart_count += 1

    def update(self, r_t, x_used):
        """
        Update after new price relatives (new day)

        :param r_t: price relatives
        :param x_used: previous portfolio
        """
        self.observed_price_relatives.append(r_t.copy())
        self.observed_portfolios.append(x_used.copy())

        self.barrons.portfolios_used.append(x_used.copy())

        self.barrons.update(r_t)

        if self._check_restart_condition():
            self._restart_barrons()

    def simulate_trading(self, price_relatives_sequence, verbose=True, verbose_days=100):
        """
        Simulate trading over a sequence of price relatives

        :param price_relatives_sequence:
        :param verbose: Print details (days, wealth, parameters)
        :param verbose_days: number of days to repeat verbose
        """
        wealth = 1.0
        daily_wealth = [1.0]

        for day, r_t in enumerate(price_relatives_sequence):
            # Get portfolio for current day
            portfolio = self.get_portfolio()
            self.global_portfolios_used.append(portfolio.copy())

            # Compute loss
            loss = -np.log(np.dot(portfolio, r_t))
            self.global_observed_loss.append(loss)

            # Update wealth
            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            if verbose and ((day + 1) % verbose_days == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}, Beta = {self.beta:.6f}, "
                      f"Portfolio = [{', '.join([f'{x:.3f}' for x in portfolio])}]")

            # Update for next day (this may trigger restart)
            self.update(r_t, portfolio)

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(self.global_portfolios_used),
            'num_days': len(price_relatives_sequence),
            'restart_count': self.restart_count,
            'final_beta': self.beta
        }