"""
Compare all portfolio strategies on the same data.

Plots all strategies together for easy comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

from benchmarks.portfolio.buy_and_hold import run_buy_and_hold
from benchmarks.portfolio.constant_rebalance_portfolio import run_crp
from benchmarks.portfolio.best_constant_rebalanced_portfolio import run_bcrp


def compare_strategies(price_relatives, stock_names=None, crp_weights=None,
                       show_plot=True, save_path=None, additional_results=None):
    """
    Compare portfolio strategies.

    :param price_relatives: Array of price relatives
    :param stock_names: List of stock names
    :param crp_weights: Weights for CRP strategy
    :param show_plot: Whether to display the plot
    :param save_path: Path to save the plot
    :param additional_results: Dict or list of dicts with custom strategy results.
                               Each dict should have:
                               - 'strategy': strategy name (str)
                               - 'daily_wealth': array of wealth values
                               - 'final_wealth': final wealth value
    """
    price_relatives = np.array(price_relatives)
    n_days, n_stocks = price_relatives.shape

    if stock_names is None:
        stock_names = [f"Stock_{i + 1}" for i in range(n_stocks)]

    # 1. Buy and Hold (uniform)
    bah_results = run_buy_and_hold(price_relatives)

    # 2. Constant Rebalanced Portfolio
    crp_results = run_crp(price_relatives, weights=crp_weights)

    # 3. Best CRP (hindsight)
    bcrp_results = run_bcrp(price_relatives, n_restarts=10)

    # Compile results
    results = {
        'BAH': bah_results,
        'CRP': crp_results,
        'BCRP': bcrp_results,
        'price_relatives': price_relatives,
        'stock_names': stock_names
    }

    # Add additional strategies if provided
    if additional_results is not None:
        # Handle single dict or list of dicts
        if isinstance(additional_results, dict):
            additional_results = [additional_results]

        for i, add_result in enumerate(additional_results):
            # Use provided strategy name or generate one
            strategy_name = add_result.get('strategy', f'Custom_{i+1}')
            results[strategy_name] = add_result

    if show_plot:
        plot_comparison(results, save_path=save_path)

    return results


def plot_comparison(results, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define base strategies with their styles
    base_strategies = {
        'BAH': {'label': 'BAH', 'marker': 'o', 'linewidth': 2},
        'CRP': {
            'label': f"CRP, W={[f'{w:.2f}' for w in results['CRP']['weights']]}",
            'marker': 's',
            'linewidth': 2
        },
        'BCRP': {
            'label': f"BCRP, W={[f'{w:.2f}' for w in results['BCRP']['best_weights']]}",
            'marker': '^',
            'linewidth': 2
        }
    }

    # Plot base strategies
    for strategy_key, style in base_strategies.items():
        if strategy_key in results:
            days = np.arange(len(results[strategy_key]['daily_wealth']))
            ax.plot(days, results[strategy_key]['daily_wealth'],
                   label=style['label'],
                   linewidth=style['linewidth'],
                   marker=style['marker'],
                   markersize=3)

    # Plot additional strategies (your custom algorithms)
    additional_markers = ['D', 'v', 'p', '*', 'X', 'P']
    marker_idx = 0

    for key, result in results.items():
        if key not in base_strategies and key not in ['price_relatives', 'stock_names']:
            days = np.arange(len(result['daily_wealth']))
            marker = additional_markers[marker_idx % len(additional_markers)]

            ax.plot(days, result['daily_wealth'],
                   label=result.get('strategy', key),
                   linewidth=2.5,
                   marker=marker,
                   markersize=4,
                   alpha=0.8)
            marker_idx += 1

    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Wealth', fontsize=12)
    ax.set_title('Portfolio Wealth Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    if save_path is None:
        plt.show()