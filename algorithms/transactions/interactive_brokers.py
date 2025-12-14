import numpy as np
import cvxpy as cp

from algorithms.transactions.cost import Costs


class InteractiveBrokersCostUSDTiered(Costs):
    """
    Dynamic version that tracks actual wealth growth over time.

    This version maintains the actual dollar wealth as it grows/shrinks,
    providing more accurate commission calculations.
    """

    def __init__(self, initial_capital):
        """
        Args:
            initial_capital: Starting dollar amount
        """
        super().__init__()
        self.initial_capital = initial_capital
        self.current_wealth_fraction = 1.0
        self.stock_prices = None

        # IB Tiered pricing parameters
        self.cost_per_share = 0.0035
        self.min_per_order = 0.35
        self.max_pct_per_order = 0.01

    def update_state(self, wealth_fraction, stock_prices):
        self.stock_prices = np.array(stock_prices)
        self.current_wealth_fraction = wealth_fraction


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
            if shares_traded[i] < 1e-8:
                continue

            trade_value = shares_traded[i] * self.stock_prices[i]

            # Per-share commission
            per_share_commission = shares_traded[i] * self.cost_per_share

            # Apply minimum
            commission = max(per_share_commission, self.min_per_order)

            # Apply 1% cap
            max_commission = self.max_pct_per_order * trade_value
            commission = min(commission, max_commission)

            total_commission += commission

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
        dollar_trades = cp.abs(new_portfolio - prev_portfolio) * current_dollar_wealth

        # Convert to shares and apply per-share cost
        # shares = dollars / price, so: cost = shares * 0.0035 = (dollars / price) * 0.0035
        share_trades = cp.multiply(dollar_trades, 1.0 / self.stock_prices)
        total_commission = self.cost_per_share * cp.sum(share_trades)

        return total_commission / current_dollar_wealth

    def gradient(self, prev_portfolio, new_portfolio):
        """Numerical gradient"""
        return self.numerical_gradient(prev_portfolio, new_portfolio)