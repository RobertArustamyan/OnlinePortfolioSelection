import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts
from algorithms.transactions.interactive_brokers import InteractiveBrokersCostUSD
from utils.data_prep import prepare_stock_data_3split

if __name__ == '__main__':
    STOCKS = ["AAPL", "NVDA", "GOOGL"]
    TRAIN_START = "2021-01-01"
    TRAIN_END = "2023-01-01"
    VAL_END = "2024-01-20"
    TEST_END = "2026-01-20"

    # Hyperparameters to test
    DELTA = 0.8
    COST_PENALTY = 1
    ALPHA = 0.0

    INITIAL_CAPITAL = 1000
    PRICING_TYPE = "fixed"

    data_dict = prepare_stock_data_3split(
        stocks=STOCKS,
        train_start_date=TRAIN_START,
        train_end_date=TRAIN_END,
        val_end_date=VAL_END,
        test_end_date=TEST_END
    )

    train_data = data_dict['train_price_relatives']
    train_prices = data_dict['train_actual_prices']
    stock_names = data_dict['stock_names']

    # Get dates for x-axis
    train_dates = data_dict.get('train_dates', None)

    cost_model = InteractiveBrokersCostUSD(
        initial_capital=INITIAL_CAPITAL,
        pricing_type=PRICING_TYPE
    )

    n_stocks = len(STOCKS)
    T = len(train_data)
    eta = (n_stocks ** 1.25) / (np.sqrt(T * np.log(n_stocks * T)))
    beta = 1.0 / (8 * (n_stocks ** 0.25) * np.sqrt(T * np.log(n_stocks * T)))

    ons = OnlineNewtonStepCosts(
        n_stocks=n_stocks,
        eta=eta,
        beta=beta,
        delta=DELTA,
        cost_model=cost_model,
        cost_penalty=COST_PENALTY,
        alpha=ALPHA,
    )

    results = ons.simulate_trading(
        train_data,
        stock_prices_sequence=train_prices,
        verbose=True,
        verbose_days=100
    )

    # Extract tracked data
    mahalanobis_history = results['mahalanobis_history']
    transaction_costs = results['transaction_costs']
    A_history = results['A_history']

    # Compute trace of A for each day
    traces = [np.trace(A) for A in A_history]

    # Compute objective (mahalanobis + penalty * cost)

    objective_history = mahalanobis_history + COST_PENALTY * transaction_costs

    # Create x-axis (dates or day numbers)
    if train_dates is not None:
        x_axis = pd.to_datetime(train_dates)
        xlabel = "Date"
    else:
        x_axis = np.arange(len(mahalanobis_history))
        xlabel = "Day"

    # Create the 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'ONS Algorithm Analysis (λ={COST_PENALTY})', fontsize=14, fontweight='bold')

    # Plot 1: Mahalanobis term vs date
    axes[0, 0].plot(x_axis, mahalanobis_history, linewidth=2, color='blue')
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel('Mahalanobis Distance')
    axes[0, 0].set_title('Mahalanobis Term vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    if train_dates is not None:
        axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Transaction cost vs date
    axes[0, 1].plot(x_axis, transaction_costs, linewidth=2, color='red')
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel('Transaction Cost')
    axes[0, 1].set_title('Transaction Cost vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    if train_dates is not None:
        axes[0, 1].tick_params(axis='x', rotation=45)

    # Plot 3: Trace of A vs date
    axes[1, 0].plot(x_axis, traces, linewidth=2, color='green')
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel('Trace(A)')
    axes[1, 0].set_title('Trace of Matrix A vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    if train_dates is not None:
        axes[1, 0].tick_params(axis='x', rotation=45)

    # Plot 4: Total objective (mahalanobis + penalty*cost) vs date
    axes[1, 1].plot(x_axis, objective_history, linewidth=2, color='purple')
    axes[1, 1].set_xlabel(xlabel)
    axes[1, 1].set_ylabel('Objective Value')
    axes[1, 1].set_title(f'Total Objective vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    if train_dates is not None:
        axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('ons_analysis_2.png', dpi=300, bbox_inches='tight')
    plt.show()
