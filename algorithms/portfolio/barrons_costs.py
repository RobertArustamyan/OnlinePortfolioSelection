import numpy as np
import cvxpy as cp


class BarronsCosts:
    def __init__(self, n_stocks, T, beta, eta, cost_model, cost_penalty=1.0,
                 alpha=0.0, improvement_threshold=0.0):
        """
        Cost-aware Barrons algorithm.

        :param n_stocks: Number of stocks
        :param T: Time horizon
        :param beta: Regularization parameter
        :param eta: Learning rate parameter
        :param cost_model: Transaction cost model
        :param cost_penalty: Weight for transaction cost in optimization
                         Higher = more conservative trading
        :param alpha: No-trade threshold. Only rebalance if portfolio change exceeds alpha.
                   Measured as L1 distance: sum(|new_weights - old_weights|)
                   0.0 (no threshold) to 0.1 (10% total change)
        :param improvement_threshold: Only trade if expected objective improvement exceeds this.
                                  Measured as reduction in Barrons objective function.
                                  Typical range: 0.0 (no threshold) to 0.01
        """
        self.n_stocks = n_stocks
        self.T = T
        self.eta = eta
        self.beta = beta
        self.cost_model = cost_model
        self.cost_penalty = cost_penalty
        self.alpha = alpha
        self.improvement_threshold = improvement_threshold
        self._check_parameters()

        self.x_min = 1 / (n_stocks * T)
        self.x_0 = np.ones(n_stocks) / n_stocks
        # Current portfolio state
        self.x_t = self.x_0.copy()
        self.prev_portfolio = self.x_0.copy()

        # Matrices for cumulative updates
        self.A = np.eye(n_stocks) * n_stocks
        self.eta_t = self.eta * np.ones(n_stocks)

        # History
        self.observed_loss = []
        self.portfolios_used = []

    def _check_parameters(self):
        """
        Checks algorithm parameters validity
        """
        if self.beta <= 0 or self.beta > 0.5:
            raise ValueError(f"Invalid beta value: {self.beta}")
        if self.eta <= 0 or self.eta > 1:
            raise ValueError(f"Invalid eta value: {self.eta}")

    def get_portfolio(self):
        """
        Return portfolio for trading.
        Applies no-trade threshold if alpha > 0.
        """
        proposed_portfolio = self.x_t.copy()

        # No-trade threshold
        if self.alpha > 0:
            portfolio_change = np.sum(np.abs(proposed_portfolio - self.prev_portfolio))

            if portfolio_change < self.alpha:
                return self.prev_portfolio.copy()

        # Improvement threshold
        if self.improvement_threshold > 0:
            # Compute current objective value (staying with prev_portfolio)
            obj_current = self._psi(self.prev_portfolio)

            # Compute proposed objective value
            obj_proposed = self._psi(proposed_portfolio)

            improvement = obj_current - obj_proposed
            if improvement < self.improvement_threshold:
                return self.prev_portfolio.copy()

        return proposed_portfolio

    def _calculate_loss(self, r_t):
        return -np.log(np.dot(self.x_t, r_t))

    def _compute_eta_t(self):
        portfolio_hist = np.array(self.portfolios_used)
        values = 1.0 / (self.n_stocks * portfolio_hist)

        log_T_values = np.log(values) / np.log(self.T)
        max_values = np.max(log_T_values, axis=0)

        return self.eta * np.exp(max_values)

    def _psi(self, x):
        x = np.asarray(x)

        first_term = (0.5 * self.beta) * (x @ self.A @ x)
        second_term = np.sum((1.0 / self.eta_t) * np.log(1.0 / x))

        return first_term + second_term

    def _grad_psi(self, x):
        x = np.asarray(x)

        first_term = self.beta * (self.A @ x)
        second_term = -1.0 / (self.eta_t * x)

        return first_term + second_term

    def _bregman_divergence(self, x, y):
        grad_y = self._grad_psi(y)
        psi_x = self._psi(x)
        psi_y = self._psi(y)

        return psi_x - psi_y - np.dot(grad_y, x - y)

    def _solve_ipm_with_costs(self, grad_t):
        """
        Solve the interior point method optimization with transaction costs.
        Minimizes: linear_term @ x + psi(x) + cost_penalty * cost(prev, x)
        """
        n = self.n_stocks
        x = cp.Variable(n)

        grad_psi_xt = self._grad_psi(self.x_t)

        # Original Barrons terms
        quad_term = 0.5 * self.beta * cp.quad_form(x, self.A)
        ent_term_lin = cp.sum(cp.multiply(-1.0 / (self.eta_t * self.x_t), x - self.x_t))
        psi_expr = quad_term + ent_term_lin

        linear_term = grad_t - grad_psi_xt

        cost_expr = self.cost_model.cvxpy_cost_linearized(prev_portfolio=self.prev_portfolio,
                                                          new_portfolio_point=self.x_t, x_var=x)

        objective = cp.Minimize(linear_term @ x + psi_expr + self.cost_penalty * cost_expr)

        constraints = [cp.sum(x) == 1, x >= self.x_min]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CVXOPT, verbose=False)

        if x.value is None:
            raise RuntimeError("Solver failed")

        x_val = np.maximum(x.value, self.x_min)
        x_val /= x_val.sum()
        return x_val

    def update(self, r_t):
        x_used = self.portfolios_used[-1]
        grad_t = -r_t / np.dot(x_used, r_t)

        self.A += np.outer(grad_t, grad_t)
        self.eta_t = self._compute_eta_t()

        self.x_t = self._solve_ipm_with_costs(grad_t)

    def simulate_trading(self, price_relatives_sequence, stock_prices_sequence=None, verbose=True, verbose_days=100):
        wealth = 1.0
        daily_wealth = [1.0]
        transaction_costs = []
        turnovers = []
        num_no_trades = 0
        num_alpha_blocks = 0
        num_improvement_blocks = 0
        block_reasons = []

        for day, r_t in enumerate(price_relatives_sequence):
            if hasattr(self.cost_model, 'update_state'):
                if stock_prices_sequence is not None:
                    day_prices = stock_prices_sequence[day]  # need checking
                else:
                    raise ValueError("Something went wrong in hasattr if statement")
                self.cost_model.update_state(wealth_fraction=wealth, stock_prices=day_prices)

            # Get portfolio for current day (with thresholds)
            portfolio = self.get_portfolio()

            actually_traded = not np.allclose(portfolio, self.prev_portfolio)
            if not actually_traded:
                num_no_trades += 1
                # Try to determine which threshold blocked
                portfolio_change = np.sum(np.abs(self.x_t - self.prev_portfolio))
                if self.alpha > 0 and portfolio_change < self.alpha:
                    num_alpha_blocks += 1
                    block_reasons.append('alpha')
                elif self.improvement_threshold > 0:
                    num_improvement_blocks += 1
                    block_reasons.append('improvement')
                else:
                    block_reasons.append('other')
            else:
                block_reasons.append('traded')

            self.portfolios_used.append(portfolio.copy())

            # Compute transaction cost for the rebalancing
            tc = self.cost_model.compute_cost(self.prev_portfolio, portfolio)

            # Apply transaction cost to wealth
            wealth *= (1 - tc)
            transaction_costs.append(tc)

            turnover = np.sum(np.abs(portfolio - self.prev_portfolio))
            turnovers.append(turnover)

            loss = self._calculate_loss(r_t)
            self.observed_loss.append(loss.copy())

            # Apply market returns
            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            # Update for next day
            self.update(r_t)
            self.prev_portfolio = portfolio.copy()

            if verbose and ((day + 1) % verbose_days == 0 or day == 0):
                print(f"Day {day + 1}: Wealth = {wealth:.4f}, "
                      f"TC = {transaction_costs[-1]:.6f}, "
                      f"Turnover = {turnover:.4f}, "
                      f"No-trades: {num_no_trades} (α:{num_alpha_blocks}, imp:{num_improvement_blocks})")

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth),
            'portfolios_used': np.array(self.portfolios_used),
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
            'block_reasons': block_reasons
        }
