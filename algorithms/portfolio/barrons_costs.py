import numpy as np
import cvxpy as cp


class BarronsCosts:
    def __init__(self, n_stocks, T, beta, eta, cost_model, cost_penalty=1.0,
                 alpha=0.0, optimization_method='linear'):
        """
        Cost-aware BARRONS algorithm.

        :param n_stocks: Number of stocks
        :param T: Time horizon
        :param beta: Regularization parameter
        :param eta: Learning rate parameter
        :param cost_model: Transaction cost model
        :param cost_penalty: Weight for transaction cost (higher = more conservative)
        :param alpha: No-trade threshold (L1 distance)
        :param optimization_method: 'linear', 'quadratic', or 'sqp'
        """
        self.n_stocks = n_stocks
        self.T = T
        self.eta = eta
        self.beta = beta
        self.cost_model = cost_model
        self.cost_penalty = cost_penalty
        self.alpha = alpha
        self.opt_method = optimization_method
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
        Validate algorithm parameters.
        """
        if self.beta <= 0 or self.beta > 0.5:
            raise ValueError(f"Invalid beta value: {self.beta}")
        if self.eta <= 0 or self.eta > 1:
            raise ValueError(f"Invalid eta value: {self.eta}")

    def get_portfolio(self):
        """
        Return portfolio for trading with no-trade threshold applied.

        :return: Portfolio weights
        """
        proposed_portfolio = self.x_t.copy()

        # No-trade threshold
        if self.alpha > 0:
            portfolio_change = np.sum(np.abs(proposed_portfolio - self.prev_portfolio))

            if portfolio_change < self.alpha:
                return self.prev_portfolio.copy()

        return proposed_portfolio

    def _calculate_loss(self, r_t):
        """
        Calculate logarithmic loss.

        :param r_t: Price relatives
        :return: Negative log return
        """
        return -np.log(np.dot(self.x_t, r_t))

    def _compute_eta_t(self):
        """
        Compute adaptive learning rates based on allocation history.

        :return: Adaptive learning rates
        """
        portfolio_hist = np.array(self.portfolios_used)
        values = 1.0 / (self.n_stocks * portfolio_hist)

        log_T_values = np.log(values) / np.log(self.T)
        max_values = np.max(log_T_values, axis=0)

        return self.eta * np.exp(max_values)

    def _psi(self, x):
        """
        Compute mixture regularizer.

        :param x: Portfolio weights
        :return: Regularizer value
        """
        x = np.asarray(x)

        first_term = (0.5 * self.beta) * (x @ self.A @ x)
        second_term = np.sum((1.0 / self.eta_t) * np.log(1.0 / x))

        return first_term + second_term

    def _grad_psi(self, x):
        """
        Compute gradient of regularizer.

        :param x: Portfolio weights
        :return: Gradient
        """
        x = np.asarray(x)

        first_term = self.beta * (self.A @ x)
        second_term = -1.0 / (self.eta_t * x)

        return first_term + second_term

    def _bregman_divergence(self, x, y):
        """
        Compute Bregman divergence.

        :param x: First portfolio
        :param y: Second portfolio
        :return: Bregman divergence
        """
        grad_y = self._grad_psi(y)
        psi_x = self._psi(x)
        psi_y = self._psi(y)

        return psi_x - psi_y - np.dot(grad_y, x - y)

    def _solve_ipm_with_costs(self, grad_t):
        """
        Solve optimization with transaction costs.
        Minimizes: linear_term @ x + psi(x) + cost_penalty * cost(prev, x)

        :param grad_t: Loss gradient
        :return: Optimal portfolio weights
        """
        if self.opt_method == 'sqp':
            return self._solve_with_sqp(grad_t)
        n = self.n_stocks
        x = cp.Variable(n)

        grad_psi_xt = self._grad_psi(self.x_t)

        # Original Barrons terms
        quad_term = 0.5 * self.beta * cp.quad_form(x, self.A)
        ent_term_lin = cp.sum(cp.multiply(-1.0 / (self.eta_t * self.x_t), x - self.x_t))
        psi_expr = quad_term + ent_term_lin

        linear_term = grad_t - grad_psi_xt

        if self.opt_method is None or self.opt_method == 'linear':
            cost_expr = self.cost_model.cvxpy_cost_linearized(prev_portfolio=self.prev_portfolio,
                                                              new_portfolio_point=self.x_t, x_var=x)
        elif self.opt_method == 'quadratic':
            cost_expr = self.cost_model.cvxpy_cost_quadratic(prev_portfolio=self.prev_portfolio,
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

    def _solve_with_sqp(self, grad_t, max_iter=7, tol=1e-4):
        """
        Sequential Quadratic Programming solver with iterative cost approximation.

        :param grad_t: Loss gradient
        :param max_iter: Maximum iterations
        :param tol: Convergence tolerance
        :return: Optimized portfolio weights
        """
        x_current = self.x_t.copy()

        for iteration in range(max_iter):
            # Build quadratic approximation at current point
            cost_value = self.cost_model.compute_cost(self.prev_portfolio, x_current)
            cost_grad = self.cost_model.numerical_gradient(self.prev_portfolio, x_current)
            cost_hessian = self.cost_model.numerical_hessian(self.prev_portfolio, x_current)

            # Solve QP subproblem
            x = cp.Variable(self.n_stocks)

            # Original Barrons terms
            grad_psi_xt = self._grad_psi(x_current)
            quad_term = 0.5 * self.beta * cp.quad_form(x, self.A)
            ent_term_lin = cp.sum(cp.multiply(-1.0 / (self.eta_t * x_current), x - x_current))
            psi_expr = quad_term + ent_term_lin

            linear_term = grad_t - grad_psi_xt

            # Quadratic cost approximation
            diff = x - x_current
            cost_expr = cost_value + cost_grad @ diff + 0.5 * cp.quad_form(diff, cost_hessian)

            objective = cp.Minimize(linear_term @ x + psi_expr + self.cost_penalty * cost_expr)
            constraints = [cp.sum(x) == 1, x >= self.x_min]

            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CVXOPT, verbose=False)

            if prob.status not in ['optimal', 'optimal_inaccurate']:
                break

            x_new = x.value
            x_new = np.maximum(x_new, self.x_min)
            x_new /= x_new.sum()

            # Check convergence
            if np.linalg.norm(x_new - x_current) < tol:
                break

            x_current = x_new

        return x_current

    def update(self, r_t):
        """
        Update algorithm state with new price relatives.

        :param r_t: Price relatives
        """
        x_used = self.portfolios_used[-1]
        grad_t = -r_t / np.dot(x_used, r_t)

        self.A += np.outer(grad_t, grad_t)
        self.eta_t = self._compute_eta_t()

        self.x_t = self._solve_ipm_with_costs(grad_t)

    def simulate_trading(self, price_relatives_sequence, stock_prices_sequence=None, verbose=True, verbose_days=100):
        """
        Simulate trading with transaction costs.

        :param price_relatives_sequence: Sequence of price relatives
        :param stock_prices_sequence: Sequence of stock prices (for cost model)
        :param verbose: Print progress
        :param verbose_days: Print frequency
        :return: Dictionary with trading results and cost statistics
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
            if hasattr(self.cost_model, 'update_state'):
                if stock_prices_sequence is not None:
                    day_prices = stock_prices_sequence[day]
                else:
                    raise ValueError("Stock prices required for cost model update")
                self.cost_model.update_state(wealth_fraction=wealth, stock_prices=day_prices)

            # Get portfolio for current day (with thresholds)
            portfolio = self.get_portfolio()

            actually_traded = not np.allclose(portfolio, self.prev_portfolio)
            if not actually_traded:
                num_no_trades += 1
                # Determine which threshold blocked
                portfolio_change = np.sum(np.abs(self.x_t - self.prev_portfolio))
                if self.alpha > 0 and portfolio_change < self.alpha:
                    num_alpha_blocks += 1
                    block_reasons.append('alpha')
                else:
                    block_reasons.append('other')
            else:
                block_reasons.append('traded')

            self.portfolios_used.append(portfolio.copy())

            # Compute transaction cost
            tc = self.cost_model.compute_cost(self.prev_portfolio, portfolio)
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