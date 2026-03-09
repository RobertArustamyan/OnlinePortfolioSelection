import numpy as np
import cvxpy as cp
from llvmlite.binding import has_svml

from algorithms.transactions.cost import Costs


class InteractiveBrokersCostUSD(Costs):
    """
    Dynamic version that tracks actual wealth growth over time.

    This version maintains the actual dollar wealth as it grows/shrinks,
    providing more accurate commission calculations.

    Updated: 01.02.2026
    """

    def __init__(self, initial_capital, pricing_type, cash_index=None):
        """
        Args:
            initial_capital: Starting dollar amount
        """
        super().__init__()
        self.initial_capital = initial_capital
        self.current_wealth_fraction = 1.0
        self.stock_prices = None
        self.pricing_type = pricing_type.lower()
        self.cash_index = cash_index

        if self.pricing_type not in ["tiered", "fixed"]:
            raise ValueError(f"pricing_type must be 'tiered' or 'fixed', got '{pricing_type}'")

        if self.pricing_type == "fixed":
            self.cost_per_share = 0.005
            self.min_per_order = 1.00
            # self.max_pct_per_trade = 0.01
            self.max_pct_per_order = 0.01
        else:
            # From Tiered version considering for now only <= 300,000 monthly volume
            self.cost_per_share = 0.0035
            self.min_per_order = 0.35
            self.max_pct_per_order = 0.005 # assuming order = trade (oder is done in one trade, so 1% changes to 0.5%)

        # Fractional share pricing
        self.fractional_min = 0.01  # $0.01 minimum for fractional shares
        self.fractional_rate = 0.01  # 1% of trade value for fractional shares

        # Maximum per order/trade
        self.max_per_trade = 9.79

    def update_state(self, wealth_fraction, stock_prices):
        self.stock_prices = np.array(stock_prices)
        self.current_wealth_fraction = wealth_fraction

    def gradient(self, prev_portfolio, new_portfolio):
        """Numerical gradient"""
        return self.numerical_gradient(prev_portfolio, new_portfolio)

    def compute_cost(self, prev_portfolio, new_portfolio):
        """
        Compute Interactive Brokers commission.

        Uses current_wealth for accurate dollar calculations.
        """
        if self.stock_prices is None:
            raise ValueError("Stock prices not set. Call update_state() first.")

        n_stocks = len(prev_portfolio)
        current_dollar_wealth = self.current_wealth_fraction * self.initial_capital

        # Convert weights to dollar values using current wealth
        prev_dollars = prev_portfolio * current_dollar_wealth
        new_dollars = new_portfolio * current_dollar_wealth

        # Convert to shares
        prev_shares = prev_dollars / self.stock_prices
        new_shares = new_dollars / self.stock_prices

        # Shares traded
        shares_traded = np.abs(new_shares - prev_shares)

        # Compute commission for each stock
        total_commission = 0.0

        for i in range(n_stocks):
            # if shares_traded[i] < 1e-8:
            #     continue
            if self.cash_index is not None and i == self.cash_index:
                continue

            whole_shares = np.floor(shares_traded[i])
            fractional_shares = shares_traded[i] - whole_shares

            stock_commission = 0.0

            if whole_shares >= 1.0:
                whole_trade_value = whole_shares * self.stock_prices[i]
                base_commission = whole_shares * self.cost_per_share
                percentage_cap = whole_trade_value * self.max_pct_per_order
                whole_commission = max(self.min_per_order, min(base_commission, percentage_cap, self.max_per_trade))

                stock_commission += whole_commission

            if fractional_shares > 0:
                fractional_trade_value = fractional_shares * self.stock_prices[i]
                fractional_commission = max(self.fractional_min, self.fractional_rate * fractional_trade_value)
                fractional_commission = min(self.max_per_trade, fractional_commission)

                stock_commission += fractional_commission

            total_commission += stock_commission

        # Return as fraction of current wealth
        return total_commission / current_dollar_wealth

    def cvxpy_cost(self, prev_portfolio, new_portfolio):
        """
        Linear approximation for convex optimization.
        """
        if self.stock_prices is None:
            raise ValueError("Stock prices not set")

        current_dollar_wealth = self.current_wealth_fraction * self.initial_capital

        # Dollar changes
        if self.cash_index is not None:
            indices = [i for i in range(len(prev_portfolio)) if i != self.cash_index]
            dollar_trades = cp.abs(new_portfolio[indices] - prev_portfolio[indices]) * current_dollar_wealth
            share_trades = cp.multiply(dollar_trades, 1.0 / self.stock_prices[indices])
        else:
            dollar_trades = cp.abs(new_portfolio - prev_portfolio) * current_dollar_wealth
            share_trades = cp.multiply(dollar_trades, 1.0 / self.stock_prices)
        total_commission = self.cost_per_share * cp.sum(share_trades)

        return total_commission / current_dollar_wealth

    def cvxpy_cost_linearized(self, prev_portfolio, new_portfolio_point, x_var):
        # Zero-order term: cost at expansion point
        cost_at_point = self.compute_cost(prev_portfolio, new_portfolio_point)

        # First-order term: gradient
        cost_grad = self.numerical_gradient(prev_portfolio, new_portfolio_point)

        # Taylor approximation
        return cost_at_point + cost_grad @ (x_var - new_portfolio_point)

    def cvxpy_cost_quadratic(self, prev_portfolio, new_portfolio_point, x_var, hessian_epsilon=1e-4):
        # Zero-order term: cost at expansion point
        cost_at_point = self.compute_cost(prev_portfolio, new_portfolio_point)

        # First-order term: gradient
        cost_grad = self.numerical_gradient(prev_portfolio, new_portfolio_point)

        # Second-order term: Hessian
        hessian = self.numerical_hessian(prev_portfolio, new_portfolio_point, epsilon=hessian_epsilon)

        # Deviation from expansion point
        diff = x_var - new_portfolio_point

        # Quadratic Taylor approximation
        quadratic_approx = (
                cost_at_point +
                cost_grad @ diff +
                0.5 * cp.quad_form(diff, hessian)
        )

        return quadratic_approx

    def numerical_hessian(self, prev_portfolio, new_portfolio, epsilon=1e-4):
        n = len(new_portfolio)
        H = np.zeros((n, n))

        eps = max(epsilon, 1e-4)

        g0 = self.numerical_gradient(prev_portfolio, new_portfolio, eps=eps / 10)

        for j in range(n):

            x_pert = new_portfolio.copy()
            x_pert[j] += eps
            x_pert[0] -= eps  # Maintain sum = 1

            # Project to feasible set (respecting lower bounds)
            x_pert = np.maximum(x_pert, 1e-8)
            x_pert = x_pert / np.sum(x_pert)  # Renormalize to ensure sum = 1

            # Compute gradient at perturbed point
            g_pert = self.numerical_gradient(prev_portfolio, x_pert, eps=eps / 10)

            # Finite difference: H[:, j] ≈ (g(x + e_j) - g(x)) / ε
            # Account for actual perturbation size (may differ due to projection)
            actual_eps = np.linalg.norm(x_pert - new_portfolio)
            if actual_eps > 1e-10:
                H[:, j] = (g_pert - g0) / actual_eps
            else:
                # If perturbation was too small, use epsilon as fallback
                H[:, j] = (g_pert - g0) / eps

        # Symmetrize the Hessian (should be symmetric theoretically)
        H = 0.5 * (H + H.T)

        # Stronger regularization for numerical stability (Fix 3)
        min_eig = np.min(np.linalg.eigvalsh(H))  # Efficient for symmetric matrices
        if min_eig < 1e-4:  # Increased threshold from 1e-6
            # Add stronger regularization to make it positive definite
            reg = max(-min_eig + 1e-4, 1e-4)  # Increased from 1e-5
            H = H + reg * np.eye(n)

        return H

