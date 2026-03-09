"""
Cost-aware Ada-BARRONS algorithm
Based on: https://arxiv.org/pdf/1805.07430
"""

import numpy as np
import cvxpy as cp

from algorithms.portfolio.barrons_costs import BarronsCosts


class AdaBarronsCosts:
    def __init__(self, n_stocks, T, cost_model, cost_penalty=1.0,
                 alpha=0.0):
        """
        Cost-aware Ada-BARRONS algorithm with adaptive beta tuning.

        :param n_stocks: Number of stocks
        :param T: Time horizon
        :param cost_model: Transaction cost model
        :param cost_penalty: Weight for transaction cost in optimization
        :param alpha: No-trade threshold
        """
        self.n_stocks = n_stocks
        self.T = T
        self.cost_model = cost_model
        self.cost_penalty = cost_penalty
        self.alpha_threshold = alpha

        # Ada-BARRONS parameters from paper
        self.eta = 1.0 / (2048 * n_stocks * (np.log(T) ** 2))
        self.beta = 0.5
        self.gamma = 0.04

        # Initialize BARRONS with current beta
        self.barrons = BarronsCosts(
            n_stocks=n_stocks,
            T=T,
            beta=self.beta,
            eta=self.eta,
            cost_model=cost_model,
            cost_penalty=cost_penalty,
            alpha=alpha,
        )

        # Global tracking across all epochs (including restarts)
        self.global_portfolios_used = []
        self.global_observed_loss = []
        self.restart_count = 0

        # Epoch-specific tracking (reset on restart)
        self.observed_price_relatives = []
        self.observed_portfolios = []

    def _compute_u_t(self):
        """
        Compute regularized leader u_t.
        """
        if len(self.observed_price_relatives) == 0:
            return np.ones(self.n_stocks) / self.n_stocks

        u = cp.Variable(self.n_stocks)

        # Stack all observed price relatives
        R = np.array(self.observed_price_relatives)

        loss_term = -cp.sum(cp.log(R @ u))

        reg_term = (1.0 / self.gamma) * cp.sum(-cp.log(u))

        objective = cp.Minimize(loss_term + reg_term)

        constraints = [cp.sum(u) == 1, u >= self.barrons.x_min]
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-6, feastol=1e-8, verbose=False)
            if u.value is None:
                raise RuntimeError("ECOS failed")
            u_val = np.maximum(u.value, self.barrons.x_min)
            u_val /= u_val.sum()
            return u_val
        except:
            try:
                prob.solve(solver=cp.CVXOPT, verbose=False)
                if u.value is None:
                    raise RuntimeError("CVXOPT failed")
                u_val = np.maximum(u.value, self.barrons.x_min)
                u_val /= u_val.sum()
                return u_val
            except:
                try:
                    prob.solve(solver=cp.SCS, eps=1e-5, max_iters=5000, verbose=False)
                    if u.value is None:
                        print("Warning: All solvers failed. Using uniform portfolio for u_t.")
                        return np.ones(self.n_stocks) / self.n_stocks
                    u_val = np.maximum(u.value, self.barrons.x_min)
                    u_val /= u_val.sum()
                    return u_val
                except:
                    print("Warning: All solvers failed with exceptions. Using uniform portfolio for u_t.")
                    return np.ones(self.n_stocks) / self.n_stocks

    def _compute_alpha_t(self, u_t):
        if len(self.observed_price_relatives) == 0:
            return 0.5

        min_alpha = 0.5

        for s in range(len(self.observed_price_relatives)):
            r_s = self.observed_price_relatives[s]
            x_s = self.observed_portfolios[s]

            delta_s = -r_s / np.dot(x_s, r_s)

            diff = u_t - x_s
            inner_product = np.dot(diff, delta_s)
            abs_inner_product = np.abs(inner_product)

            # Avoid division by zero
            if abs_inner_product > 1e-10:
                alpha_s = 1.0 / (8.0 * abs_inner_product)
                min_alpha = min(min_alpha, alpha_s)

        return min_alpha

    def get_portfolio(self):
        """
        Return portfolio for trading.
        Delegates to underlying BARRONS instance.
        """
        return self.barrons.get_portfolio()

    def _check_restart_condition(self):
        if len(self.observed_price_relatives) == 0:
            return False

        u_t = self._compute_u_t()
        alpha_t = self._compute_alpha_t(u_t)

        return self.beta > alpha_t

    def _restart_barrons(self):
        self.beta = self.beta / 2.0

        # Create new BARRONS instance with updated beta
        self.barrons = BarronsCosts(
            n_stocks=self.n_stocks,
            T=self.T,
            beta=self.beta,
            eta=self.eta,
            cost_model=self.cost_model,
            cost_penalty=self.cost_penalty,
            alpha=self.alpha_threshold,
        )

        # Reset epoch-specific tracking
        self.observed_price_relatives = []
        self.observed_portfolios = []
        self.restart_count += 1

    def update(self, r_t, x_used):
        """
        Update algorithm after observing price relatives.

        :param r_t: Price relatives vector
        :param x_used: Portfolio used for this period
        """
        # Track for restart condition checking
        self.observed_price_relatives.append(r_t.copy())
        self.observed_portfolios.append(x_used.copy())

        # Update underlying BARRONS
        # Note: We manually add to portfolios_used because BARRONS expects
        # this to be before update() is called
        self.barrons.portfolios_used.append(x_used.copy())
        self.barrons.update(r_t)

        # Check if restart is needed
        if self._check_restart_condition():
            self._restart_barrons()

    def simulate_trading(self, price_relatives_sequence, stock_prices_sequence=None,
                         verbose=True, verbose_days=100):
        """
        Simulate trading with transaction costs over a sequence of price relatives.

        :param price_relatives_sequence: Array of price relative vectors
        :param stock_prices_sequence: Array of stock prices (for cost models that need them)
        :param verbose: Whether to print progress
        :param verbose_days: How often to print progress
        :return: Dictionary with trading results
        """
        wealth = 1.0
        daily_wealth = [1.0]
        transaction_costs = []
        turnovers = []
        num_no_trades = 0
        num_alpha_blocks = 0
        num_improvement_blocks = 0
        block_reasons = []

        for day, r_t in enumerate(price_relatives_sequence):
            # Update cost model state if needed
            if hasattr(self.cost_model, 'update_state'):
                if stock_prices_sequence is not None:
                    day_prices = stock_prices_sequence[day]
                else:
                    day_prices = None
                self.cost_model.update_state(wealth_fraction=wealth, stock_prices=day_prices)

            # Get portfolio for current day (with thresholds from BARRONS)
            portfolio = self.get_portfolio()
            prev_portfolio = self.barrons.prev_portfolio.copy()

            # Track if we actually traded
            actually_traded = not np.allclose(portfolio, prev_portfolio)
            if not actually_traded:
                num_no_trades += 1
                portfolio_change = np.sum(np.abs(self.barrons.x_t - prev_portfolio))
                if self.alpha_threshold > 0 and portfolio_change < self.alpha_threshold:
                    num_alpha_blocks += 1
                    block_reasons.append('alpha')
                else:
                    block_reasons.append('other')
            else:
                block_reasons.append('traded')

            # Store portfolio
            self.global_portfolios_used.append(portfolio.copy())

            # Compute transaction cost
            tc = self.cost_model.compute_cost(prev_portfolio, portfolio)
            wealth *= (1 - tc)
            transaction_costs.append(tc)

            # Compute turnover
            turnover = np.sum(np.abs(portfolio - prev_portfolio))
            turnovers.append(turnover)

            # Compute and store loss
            loss = -np.log(np.dot(portfolio, r_t))
            self.global_observed_loss.append(loss)

            # Apply market returns
            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            if verbose and ((day + 1) % verbose_days == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}, Beta = {self.beta:.6f}, "
                      f"TC = {transaction_costs[-1]:.6f}, Turnover = {turnover:.4f}, "
                      f"Restarts: {self.restart_count}")

            # Update for next day (may trigger restart)
            self.update(r_t, portfolio)

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(self.global_portfolios_used),
            'transaction_costs': np.array(transaction_costs),
            'turnovers': np.array(turnovers),
            'total_transaction_cost': sum(transaction_costs),
            'avg_turnover': np.mean(turnovers),
            'max_turnover': np.max(turnovers),
            'num_days': len(price_relatives_sequence),
            'num_no_trades': num_no_trades,
            'num_alpha_blocks': num_alpha_blocks,
            'num_improvement_blocks': num_improvement_blocks,
            'trade_frequency': 1 - (num_no_trades / len(price_relatives_sequence)),
            'block_reasons': block_reasons,
            'restart_count': self.restart_count,
            'final_beta': self.beta
        }