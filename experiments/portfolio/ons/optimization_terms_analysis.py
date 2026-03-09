import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent))

from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts
from algorithms.transactions.interactive_brokers import InteractiveBrokersCostUSD
from utils.data_prep import prepare_stock_data_3split


class ObjectiveTermAnalyzer:
    """Analyze objective function components during ONS trading"""

    def __init__(self, ons_algorithm, cost_model, stock_names):
        self.ons = ons_algorithm
        self.cost_model = cost_model
        self.stock_names = stock_names

        # Tracking arrays
        self.mahalanobis_terms = []
        self.cost_terms = []
        self.cost_terms_with_penalty = []
        self.portfolio_updates = []
        self.eigenvalues_history = []
        self.traces = []
        self.condition_numbers = []
        self.target_portfolios = []
        self.actual_portfolios = []

    def track_iteration(self, day, portfolio_used, prev_portfolio, price_relatives):
        """Track all metrics for one trading day"""

        # Get current A and b
        A_t = self.ons.A.copy()
        b_t = self.ons.b.copy()

        # Compute target portfolio
        try:
            A_inv = np.linalg.inv(A_t)
            x_t = self.ons.delta * A_inv @ b_t
        except np.linalg.LinAlgError:
            x_t = portfolio_used  # Fallback

        self.target_portfolios.append(x_t.copy())
        self.actual_portfolios.append(portfolio_used.copy())

        # 1. Mahalanobis distance term
        diff = portfolio_used - x_t
        mahalanobis = diff @ A_t @ diff
        self.mahalanobis_terms.append(mahalanobis)

        # 2. Cost term (without penalty)
        cost = self.cost_model.compute_cost(prev_portfolio, portfolio_used)
        self.cost_terms.append(cost)

        # 3. Cost term with penalty
        cost_with_penalty = self.ons.cost_penalty * cost
        self.cost_terms_with_penalty.append(cost_with_penalty)

        # 4. Portfolio update size
        if day > 0:
            update_size = np.linalg.norm(portfolio_used - prev_portfolio)
            self.portfolio_updates.append(update_size)

        # 5. A matrix properties
        eigenvalues = np.linalg.eigvalsh(A_t)
        self.eigenvalues_history.append(eigenvalues)
        self.traces.append(np.trace(A_t))
        self.condition_numbers.append(np.linalg.cond(A_t))

    def simulate_with_tracking(self, price_relatives_sequence, stock_prices_sequence):
        """Run simulation and track all metrics"""

        wealth = 1.0
        daily_wealth = [1.0]

        print(f"Simulating {len(price_relatives_sequence)} days with tracking...\n")

        for day, r_t in enumerate(price_relatives_sequence):
            # Update cost model state
            if hasattr(self.cost_model, 'update_state'):
                self.cost_model.update_state(
                    wealth_fraction=wealth,
                    stock_prices=stock_prices_sequence[day]
                )

            # Get portfolio
            portfolio = self.ons.get_portfolio()
            prev_portfolio = self.ons.prev_portfolio.copy()

            # Track metrics BEFORE update
            self.track_iteration(day, portfolio, prev_portfolio, r_t)

            # Compute transaction cost
            tc = self.cost_model.compute_cost(prev_portfolio, portfolio)
            wealth *= (1 - tc)

            # Apply returns
            daily_return = np.dot(portfolio, r_t)
            wealth *= daily_return
            daily_wealth.append(wealth)

            # Update algorithm
            self.ons.update(r_t, portfolio)

            # Progress
            if (day + 1) % 100 == 0:
                print(f"Day {day + 1}/{len(price_relatives_sequence)} complete")

        print(f"\nSimulation complete. Final wealth: {wealth:.4f}\n")

        return {
            'final_wealth': wealth,
            'daily_wealth': np.array(daily_wealth)
        }

    def plot_analysis(self, save_path_prefix=None):
        """Create separate plots for objective components (no log scale, no fixed λ display)"""

        days = np.arange(len(self.mahalanobis_terms))

        mahal = np.array(self.mahalanobis_terms)
        cost = np.array(self.cost_terms)
        cost_pen = np.array(self.cost_terms_with_penalty)

        # ==============================
        # 1. Cost term only
        # ==============================
        plt.figure(figsize=(8, 4))
        plt.plot(days, cost, linewidth=2)
        plt.xlabel("Day")
        plt.ylabel("Transaction Cost")
        plt.title("Transaction Cost Term")
        plt.grid(True, alpha=0.3)

        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_cost.png", dpi=300, bbox_inches="tight")
        plt.show()

        # ==============================
        # 2. Mahalanobis term only
        # ==============================
        plt.figure(figsize=(8, 4))
        plt.plot(days, mahal, linewidth=2)
        plt.xlabel("Day")
        plt.ylabel("Mahalanobis Term")
        plt.title("Mahalanobis Distance Term")
        plt.grid(True, alpha=0.3)

        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_mahalanobis.png", dpi=300, bbox_inches="tight")
        plt.show()

        # ==============================
        # 3. Combined objective terms
        # ==============================
        plt.figure(figsize=(8, 4))
        plt.plot(days, mahal, label="Mahalanobis term", linewidth=2)
        plt.plot(days, cost_pen, label="Cost term (penalized)", linewidth=2)
        plt.xlabel("Day")
        plt.ylabel("Objective Value")
        plt.title("Objective Terms (Combined)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_combined.png", dpi=300, bbox_inches="tight")
        plt.show()

        # ==============================
        # 4. Portfolio update size
        # ==============================
        if len(self.portfolio_updates) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(days[1:], self.portfolio_updates, linewidth=2)
            plt.xlabel("Day")
            plt.ylabel(r"$\|p_t - p_{t-1}\|$")
            plt.title("Portfolio Update Size")
            plt.grid(True, alpha=0.3)

            if save_path_prefix:
                plt.savefig(f"{save_path_prefix}_updates.png", dpi=300, bbox_inches="tight")
            plt.show()

    def print_summary(self):
        """Print numerical summary"""

        print("=" * 80)
        print("OBJECTIVE TERMS ANALYSIS")
        print("=" * 80)

        mahal_arr = np.array(self.mahalanobis_terms)
        cost_arr = np.array(self.cost_terms)
        ratio_arr = mahal_arr / np.maximum(cost_arr, 1e-10)

        print(f"\nMahalanobis Term:")
        print(f"  Initial: {mahal_arr[0]:.4e}")
        print(f"  Final:   {mahal_arr[-1]:.4e}")
        print(f"  Mean:    {np.mean(mahal_arr):.4e}")
        print(f"  Std:     {np.std(mahal_arr):.4e}")
        print(f"  Growth:  {mahal_arr[-1] / mahal_arr[0]:.2f}x")

        print(f"\nCost Term (without penalty):")
        print(f"  Initial: {cost_arr[0]:.4e}")
        print(f"  Final:   {cost_arr[-1]:.4e}")
        print(f"  Mean:    {np.mean(cost_arr):.4e}")
        print(f"  Std:     {np.std(cost_arr):.4e}")

        print(f"\nRatio (Mahalanobis / Cost):")
        print(f"  Initial:  {ratio_arr[0]:.4e}")
        print(f"  Final:    {ratio_arr[-1]:.4e}")
        print(f"  Median:   {np.median(ratio_arr):.4e}")
        print(f"  Growth:   {ratio_arr[-1] / ratio_arr[0]:.2f}x")

        print(f"\nCurrent Cost Penalty: λ = {self.ons.cost_penalty}")
        print(f"Suggested λ for balance: {np.median(ratio_arr):.4e}")

        print(f"\nA Matrix (final):")
        eigs_final = self.eigenvalues_history[-1]
        print(f"  Eigenvalues: {eigs_final}")
        print(f"  Trace: {self.traces[-1]:.2f}")
        print(f"  Condition: {self.condition_numbers[-1]:.2e}")



def run_analysis(stocks, train_start, train_end, val_end, test_end,
                 delta, cost_penalty, alpha, improvement_threshold,
                 initial_capital, pricing_type="fixed",
                 save_plot=True, plot_path="objective_analysis.png"):
    """
    Run complete analysis for one configuration

    Args:
        stocks: List of stock tickers
        train_start, train_end, val_end, test_end: Date strings
        delta, cost_penalty, alpha, improvement_threshold: ONS hyperparameters
        initial_capital: Starting capital
        pricing_type: "fixed" or "tiered"
        save_plot: Whether to save plot
        plot_path: Path to save plot
    """

    print("=" * 80)
    print("OBJECTIVE TERMS TRACKING ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Stocks: {stocks}")
    print(f"  delta: {delta}")
    print(f"  cost_penalty (λ): {cost_penalty}")
    print(f"  alpha: {alpha}")
    print(f"  improvement_threshold: {improvement_threshold}")
    print(f"  Initial capital: ${initial_capital}")
    print(f"  Pricing: {pricing_type}")
    print()

    # Prepare data
    print("Loading data...")
    data_dict = prepare_stock_data_3split(
        stocks=stocks,
        train_start_date=train_start,
        train_end_date=train_end,
        val_end_date=val_end,
        test_end_date=test_end
    )

    # Use only training data for analysis
    price_relatives = data_dict['train_price_relatives']
    stock_prices = data_dict['train_actual_prices']
    stock_names = data_dict['stock_names']

    print(f"Data loaded: {len(price_relatives)} training days\n")

    # Create cost model
    cost_model = InteractiveBrokersCostUSD(
        initial_capital=initial_capital,
        pricing_type=pricing_type
    )

    # Create ONS algorithm
    n_stocks = len(stocks)
    T = len(price_relatives)
    eta = (n_stocks ** 1.25) / (np.sqrt(T * np.log(n_stocks * T)))
    beta = 1.0 / (8 * (n_stocks ** 0.25) * np.sqrt(T * np.log(n_stocks * T)))

    ons = OnlineNewtonStepCosts(
        n_stocks=n_stocks,
        eta=eta,
        beta=beta,
        delta=delta,
        cost_model=cost_model,
        cost_penalty=cost_penalty,
        alpha=alpha,
        improvement_threshold=improvement_threshold
    )

    print(f"ONS parameters: η={eta:.6f}, β={beta:.6f}\n")

    # Create analyzer
    analyzer = ObjectiveTermAnalyzer(ons, cost_model, stock_names)

    # Run simulation with tracking
    results = analyzer.simulate_with_tracking(price_relatives, stock_prices)

    # Print summary
    analyzer.print_summary()

    # Plot results
    analyzer.plot_analysis(save_path_prefix="objective_analysis")

    return analyzer, results


if __name__ == "__main__":
    # Configuration
    STOCKS = ["CRM", "ADBE", "ORCL"]
    TRAIN_START = "2021-01-01"
    TRAIN_END = "2023-01-01"
    VAL_END = "2024-01-20"
    TEST_END = "2026-01-20"

    # Hyperparameters to test
    DELTA = 0.8
    COST_PENALTY = 1e15 # Start with λ=1 to see natural ratio
    ALPHA = 0.0
    IMPROVEMENT_THRESHOLD = 0.0

    INITIAL_CAPITAL = 1
    PRICING_TYPE = "fixed"

    # Run analysis
    analyzer, results = run_analysis(
        stocks=STOCKS,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        val_end=VAL_END,
        test_end=TEST_END,
        delta=DELTA,
        cost_penalty=COST_PENALTY,
        alpha=ALPHA,
        improvement_threshold=IMPROVEMENT_THRESHOLD,
        initial_capital=INITIAL_CAPITAL,
        pricing_type=PRICING_TYPE,
        save_plot=True,
        plot_path="objective_analysis.png"
    )

    print("\n✓ Analysis complete!")
    print("Check 'objective_analysis.png' for visualizations")