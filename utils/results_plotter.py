from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors


class ResultsPlotter:
    """Class for creating performance analysis plots for portfolio experiments"""

    def __init__(self, results, model_name, detailed_results=None, test_data=None):
        """
        Initialize the plotter with experiment results

        results: List of result dictionaries from experiments
        model_name: Name of the cost model
        detailed_results: Detailed results for each experiment
        test_data: Test price relatives data for computing B&H trajectories
        """
        self.results = results
        self.model_name = model_name
        self.df = pd.DataFrame(results)
        self.df['model'] = model_name
        self.detailed_results = detailed_results
        self.test_data = test_data

    def plot_parameter_analysis(self, save_path=None, bah_results=None, nasdaq_bah_results=None,
                                sp500_bah_results=None):
        """
        Create parameter analysis plots (2x2 grid) - renamed from plot_all

        save_path: Path to save the plot (optional)
        bah_results: Buy and hold daily wealth array for comparison (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.model_name} - Performance Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Top 5 Wealth Trajectories
        self._plot_wealth_trajectories(axes[0, 0], bah_results, nasdaq_bah_results, sp500_bah_results)

        # Plot 2: Cost Penalty Analysis
        self._plot_cost_penalty_effect(axes[0, 1])

        # Plot 3: Alpha Analysis
        self._plot_alpha_effect(axes[1, 0])

        # Plot 4: Improvement Threshold Analysis
        self._plot_improvement_threshold_effect(axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter analysis plot saved to: {save_path}")

        plt.show()
        plt.close()

    def plot_trading_value_comparison(self, save_path=None, bah_results=None,
                                      nasdaq_bah_results=None, sp500_bah_results=None):
        """
        Create comprehensive comparison plot showing:
        - Top 5 wealth trajectories WITH trades
        - Same 5 configurations WITHOUT trades (B&H from initial portfolio)
        - Uniform B&H, NASDAQ, and S&P 500 benchmarks

        Total: 10 strategy lines + 3 benchmark lines = 13 lines

        save_path: Path to save the plot (optional)
        bah_results: Uniform buy and hold daily wealth array
        nasdaq_bah_results: NASDAQ buy and hold daily wealth array
        sp500_bah_results: S&P 500 buy and hold daily wealth array
        """
        if self.test_data is None:
            raise ValueError("test_data is required for computing B&H trajectories. "
                             "Initialize ResultsPlotter with test_data parameter.")

        fig, ax = plt.subplots(1, 1, figsize=(14, 9))
        fig.suptitle(f'{self.model_name} - Trading Value Analysis', fontsize=16, fontweight='bold')

        # Get top 5 results
        df_sorted = self.df.sort_values('test_final_wealth', ascending=False)
        top_5 = df_sorted.head(5)

        # Maximally distinct colors for 5 pairs - using well-separated colors
        # These are from ColorBrewer's qualitative Set1 palette, optimized for distinction
        strategy_colors = [
            '#e41a1c',  # Bright Red
            '#377eb8',  # Medium Blue
            '#4daf4a',  # Green
            '#ff7f00',  # Orange
            '#984ea3',  # Purple
        ]

        # Plot each of the top 5 configurations
        for idx, (_, row) in enumerate(top_5.iterrows()):
            # Configuration label
            config_label = f"CP={row['cost_penalty']:.1f}, α={row['alpha']:.3f}"

            # Plot WITH trades (solid line, thick)
            ax.plot(row['test_daily_wealth'],
                    label=f"#{idx + 1} With Trades: {config_label}",
                    linestyle='-',
                    linewidth=3.0,
                    color=strategy_colors[idx],
                    alpha=0.95)

            # Find corresponding detailed result to get initial portfolio
            detail = self._find_detailed_result(row)
            if detail:
                initial_portfolio = detail['testing']['portfolios_used'][0]

                # Compute B&H trajectory from this initial portfolio
                bh_trajectory = self._compute_bah_trajectory(initial_portfolio)

                # Plot WITHOUT trades (dashed line, same color, slightly thinner)
                ax.plot(bh_trajectory,
                        label=f"#{idx + 1} No Trades: {config_label}",
                        linestyle='--',
                        linewidth=2.5,
                        color=strategy_colors[idx],
                        alpha=0.7)

        # Plot benchmarks with distinct styles using neutral colors
        if bah_results is not None:
            ax.plot(bah_results, label='Buy & Hold (Uniform)',
                    linestyle=':', linewidth=3.5, color='black', alpha=0.9)

        if nasdaq_bah_results is not None:
            ax.plot(nasdaq_bah_results, label='NASDAQ (Buy & Hold)',
                    linestyle='-.', linewidth=3.0, color='dimgray', alpha=0.8)

        if sp500_bah_results is not None:
            ax.plot(sp500_bah_results, label='S&P 500 (Buy & Hold)',
                    linestyle=(0, (3, 1, 1, 1)), linewidth=3.0, color='gray', alpha=0.8)

        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Wealth', fontsize=12)
        ax.set_title('Top 5 Strategies: With vs Without Trading', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trading value comparison plot saved to: {save_path}")

        plt.show()
        plt.close()

    def _find_detailed_result(self, row):
        """Find the detailed result matching the given row parameters"""
        if self.detailed_results is None:
            return None

        for detail in self.detailed_results:
            params = detail['parameters']
            if (params['cost_penalty'] == row['cost_penalty'] and
                    params['alpha'] == row['alpha'] and
                    params['improvement_threshold'] == row['improvement_threshold']):
                return detail
        return None

    def _compute_bah_trajectory(self, initial_portfolio):
        """
        Compute buy-and-hold wealth trajectory from given initial portfolio

        initial_portfolio: Starting portfolio weights
        """
        wealth = 1.0
        trajectory = [wealth]

        for price_rel in self.test_data:
            daily_return = np.dot(initial_portfolio, price_rel)
            wealth *= daily_return
            trajectory.append(wealth)

        return np.array(trajectory)

    def _plot_wealth_trajectories(self, ax, bah_results=None, nasdaq_bah_results=None, sp500_bah_results=None):
        """Plot top 5 wealth trajectories"""
        df_sorted = self.df.sort_values('test_final_wealth', ascending=False)
        top_5 = df_sorted.head(5)

        for idx, (_, row) in enumerate(top_5.iterrows()):
            label = f"CP={row['cost_penalty']:.1f}, α={row['alpha']:.3f}, IT={row['improvement_threshold']:.4f}"
            linestyle = '-' if idx < 3 else '--'
            ax.plot(row['test_daily_wealth'], label=label, linestyle=linestyle, linewidth=2)

        # Plot benchmarks with distinct styles
        if bah_results is not None:
            ax.plot(bah_results, label='Buy & Hold (Portfolio)',
                    linestyle=':', linewidth=2.5, color='black', alpha=0.7)

        if nasdaq_bah_results is not None:
            ax.plot(nasdaq_bah_results, label='NASDAQ (Buy & Hold)',
                    linestyle='-.', linewidth=2.5, color='blue', alpha=0.6)

        if sp500_bah_results is not None:
            ax.plot(sp500_bah_results, label='S&P 500 (Buy & Hold)',
                    linestyle='--', linewidth=2.5, color='green', alpha=0.6)

        ax.set_xlabel('Trading Days', fontsize=11)
        ax.set_ylabel('Wealth', fontsize=11)
        ax.set_title('Top 5 Wealth Trajectories', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_cost_penalty_effect(self, ax):
        """Plot effect of cost penalty on performance"""
        ax_twin = ax.twinx()

        cost_penalty_groups = self.df.groupby('cost_penalty').agg({
            'test_final_wealth': 'mean',
            'test_trade_frequency': 'mean'
        }).reset_index()

        color1 = 'tab:blue'
        ax.scatter(cost_penalty_groups['cost_penalty'], cost_penalty_groups['test_final_wealth'],
                   color=color1, s=100, label='Final Wealth', zorder=3)
        ax.set_xlabel('Cost Penalty', fontsize=11)
        ax.set_ylabel('Test Final Wealth', color=color1, fontsize=11)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(True, alpha=0.3)

        color2 = 'tab:orange'
        ax_twin.scatter(cost_penalty_groups['cost_penalty'], cost_penalty_groups['test_trade_frequency'],
                        color=color2, s=100, marker='s', label='Trade Frequency', zorder=3)
        ax_twin.set_ylabel('Trade Frequency', color=color2, fontsize=11)
        ax_twin.tick_params(axis='y', labelcolor=color2)

        ax.set_title('Effect of Cost Penalty', fontsize=12, fontweight='bold')

    def _plot_alpha_effect(self, ax):
        """Plot effect of alpha on performance"""
        ax_twin = ax.twinx()

        alpha_groups = self.df.groupby('alpha').agg({
            'test_final_wealth': 'mean',
            'test_trade_frequency': 'mean'
        }).reset_index()

        color1 = 'tab:green'
        ax.scatter(alpha_groups['alpha'], alpha_groups['test_final_wealth'],
                   color=color1, s=100, label='Final Wealth', zorder=3)
        ax.set_xlabel('Alpha (No-Trade Threshold)', fontsize=11)
        ax.set_ylabel('Test Final Wealth', color=color1, fontsize=11)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(True, alpha=0.3)

        color2 = 'tab:red'
        ax_twin.scatter(alpha_groups['alpha'], alpha_groups['test_trade_frequency'],
                        color=color2, s=100, marker='s', label='Trade Frequency', zorder=3)
        ax_twin.set_ylabel('Trade Frequency', color=color2, fontsize=11)
        ax_twin.tick_params(axis='y', labelcolor=color2)

        ax.set_title('Effect of Alpha (No-Trade Threshold)', fontsize=12, fontweight='bold')

    def _plot_improvement_threshold_effect(self, ax):
        """Plot effect of improvement threshold on performance"""
        ax_twin = ax.twinx()

        it_groups = self.df.groupby('improvement_threshold').agg({
            'test_final_wealth': 'mean',
            'test_trade_frequency': 'mean'
        }).reset_index()

        color1 = 'tab:purple'
        ax.scatter(it_groups['improvement_threshold'], it_groups['test_final_wealth'],
                   color=color1, s=100, label='Final Wealth', zorder=3)
        ax.set_xlabel('Improvement Threshold', fontsize=11)
        ax.set_ylabel('Test Final Wealth', color=color1, fontsize=11)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(True, alpha=0.3)

        color2 = 'tab:brown'
        ax_twin.scatter(it_groups['improvement_threshold'], it_groups['test_trade_frequency'],
                        color=color2, s=100, marker='s', label='Trade Frequency', zorder=3)
        ax_twin.set_ylabel('Trade Frequency', color=color2, fontsize=11)
        ax_twin.tick_params(axis='y', labelcolor=color2)

        ax.set_title('Effect of Improvement Threshold', fontsize=12, fontweight='bold')

    def get_dataframe(self):
        """Return the results as a DataFrame"""
        return self.df

    def plot_wealth_comparison(self, save_path=None, bah_results=None, modified_bah_results=None, trade_dates=None,
                               trade_costs_dollars=None, nasdaq_bah_results=None, sp500_bah_results=None):
        """
        Create a single plot comparing best trajectory with regular and modified B&H

        save_path: Path to save the plot (optional)
        bah_results: Regular buy and hold daily wealth array (uniform start)
        modified_bah_results: List of modified B&H results (from best configuration)
        trade_dates: List of indices where trades occurred
        trade_costs_dollars: List of transaction costs in dollars corresponding to trade_dates
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'{self.model_name} - Wealth Trajectories Comparison', fontsize=16, fontweight='bold')

        df_sorted = self.df.sort_values('test_final_wealth', ascending=False)
        best_result = df_sorted.iloc[0]

        # Plot best algorithm trajectory
        label = f"Best: CP={best_result['cost_penalty']:.1f}, α={best_result['alpha']:.3f}, IT={best_result['improvement_threshold']:.4f}"
        ax.plot(best_result['test_daily_wealth'], label=label, linestyle='-', linewidth=2.5, color='tab:blue')

        # Plot regular B&H (uniform start)
        if bah_results is not None:
            ax.plot(bah_results, label='Buy & Hold (Uniform)', linestyle=':', linewidth=2.5, color='black', alpha=0.7)
        # Plot NASDAQ B&H
        if nasdaq_bah_results is not None:
            ax.plot(nasdaq_bah_results, label='NASDAQ (Buy & Hold)',
                    linestyle='-.', linewidth=2.5, color='blue', alpha=0.6)
        # Plot S&P500 B&H
        if sp500_bah_results is not None:
            ax.plot(sp500_bah_results, label='S&P 500 (Buy & Hold)',
                    linestyle='--', linewidth=2.5, color='green', alpha=0.6)

        # Plot modified B&H trajectory (matching initial portfolio)
        if modified_bah_results is not None and len(modified_bah_results) > 0:
            ax.plot(modified_bah_results[0], label='Buy & Hold (Best Config Start)',
                    linestyle='-.', linewidth=2.5, color='tab:orange', alpha=0.8)

        # Add vertical lines for trade dates with color proportional to cost
        if trade_dates is not None and len(trade_dates) > 0:
            y_min, y_max = ax.get_ylim()

            segments = [[(idx, y_min), (idx, y_max)] for idx in trade_dates]

            if trade_costs_dollars is not None and len(trade_costs_dollars) == len(trade_dates):
                costs = np.array(trade_costs_dollars)

                # Normalize costs to [0, 1]
                norm = colors.Normalize(vmin=costs.min(), vmax=costs.max())

                # Greys colormap: low → light gray, high → black
                cmap = plt.cm.Greys
                line_colors = cmap(norm(costs))

                lc = LineCollection(
                    segments,
                    colors=line_colors,
                    linewidths=1.2,
                    linestyles='dashed',
                    alpha=0.8,
                    zorder=1
                )
                ax.add_collection(lc)

                # Colorbar to explain cost → color mapping
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, pad=0.01)
                cbar.set_label('Transaction Cost ($)', fontsize=10)

                # Legend proxy
                ax.plot([], [], linestyle='--', color='gray', linewidth=1.2,
                        label=f'Trade Events ({len(trade_dates)})')

            else:
                # Fallback: uniform gray if costs unavailable
                lc = LineCollection(
                    segments,
                    colors='gray',
                    linewidths=1.2,
                    linestyles='dashed',
                    alpha=0.4,
                    zorder=1
                )
                ax.add_collection(lc)

                ax.plot([], [], linestyle='--', color='gray', linewidth=1.2,
                        label=f'Trade Events ({len(trade_dates)})')

        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Wealth', fontsize=12)
        ax.set_title('Best Algorithm vs Buy & Hold Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")

        plt.show()
        plt.close()

    def plot_portfolio(self, index, ticker_names, save_path=None):
        """
        Visualize portfolio allocation at a specific day using polar chart

        index: Day index to visualize
        ticker_names: List of ticker names corresponding to portfolio positions
        save_path: Path to save the plot (optional)
        """
        if self.detailed_results is None:
            raise ValueError("No detailed results available. Initialize ResultsPlotter with detailed_results.")

        portfolio_to_plot = self.detailed_results['testing']['portfolios_used'][index]

        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, projection='polar')

        theta = np.linspace(0, 2 * np.pi, len(ticker_names), endpoint=False)
        width = 2 * np.pi / len(ticker_names)
        colors = plt.cm.Set3(np.linspace(0, 1, len(ticker_names)))

        bars = ax.bar(theta, portfolio_to_plot, width=width, color=colors,
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xticks(theta)
        ax.set_xticklabels(ticker_names, fontsize=11, weight='bold')
        ax.set_ylim(0, max(portfolio_to_plot) * 1.2)
        ax.set_title(f'Portfolio Allocation - Day {index}\n{self.model_name}',
                     fontsize=14, fontweight='bold', pad=20)

        # Add percentage labels
        for angle, weight, ticker in zip(theta, portfolio_to_plot, ticker_names):
            ax.text(angle, weight + max(portfolio_to_plot) * 0.05,
                    f'{weight * 100:.1f}%',
                    ha='center', va='bottom', fontsize=9, weight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio visualization saved to: {save_path}")

        plt.show()
        plt.close()

    def plot_portfolio_evolution(self, ticker_names, save_path=None, sample_days=None):
        """
        Visualize how portfolio allocation changes over time
        Shows multiple polar charts in one figure

        ticker_names: List of ticker names
        save_path: Path to save the plot
        sample_days: List of specific days to plot
        """
        if self.detailed_results is None:
            raise ValueError("No detailed results available.")

        portfolios = self.detailed_results['testing']['portfolios_used']
        n_days = len(portfolios)

        # Sample days if not specified
        if sample_days is None:
            if n_days <= 10:
                sample_days = list(range(n_days))
            else:
                sample_days = [0, n_days // 4, n_days // 2, 3 * n_days // 4, n_days - 1]

        n_samples = len(sample_days)
        fig = plt.figure(figsize=(5 * n_samples, 5))

        colors = plt.cm.Set3(np.linspace(0, 1, len(ticker_names)))

        theta = np.linspace(0, 2 * np.pi, len(ticker_names), endpoint=False)
        width = 2 * np.pi / len(ticker_names)

        for idx, day in enumerate(sample_days, 1):
            ax = fig.add_subplot(1, n_samples, idx, projection='polar')
            portfolio = portfolios[day]

            bars = ax.bar(theta, portfolio, width=width, color=colors,
                          alpha=0.8, edgecolor='black', linewidth=1.5)

            ax.set_xticks(theta)
            ax.set_xticklabels(ticker_names, fontsize=11, weight='bold')
            ax.set_ylim(0, max(portfolio) * 1.2)
            ax.set_title(f'Day {day}', fontsize=12, fontweight='bold')

            # Add percentage labels
            for angle, weight in zip(theta, portfolio):
                ax.text(angle, weight + max(portfolio) * 0.05,
                        f'{weight * 100:.1f}%',
                        ha='center', va='bottom', fontsize=9, weight='bold')

        fig.suptitle(f'Portfolio Evolution - {self.model_name}',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio evolution plot saved to: {save_path}")

        plt.show()
        plt.close()