from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors


class ResultsPlotter:
    """Class for creating performance analysis plots for portfolio experiments"""

    def __init__(self, results, model_name):
        """
        Initialize the plotter with experiment results

        Args:
            results: List of result dictionaries from experiments
            model_name: Name of the cost model
        """
        self.results = results
        self.model_name = model_name
        self.df = pd.DataFrame(results)
        self.df['model'] = model_name

    def plot_all(self, save_path=None, bah_results=None):
        """
        Create all plots (2x2 grid)

        Args:
            save_path: Path to save the plot (optional)
            bah_results: Buy and hold daily wealth array for comparison (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.model_name} - Performance Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Top 5 Wealth Trajectories
        self._plot_wealth_trajectories(axes[0, 0], bah_results)

        # Plot 2: Cost Penalty Analysis
        self._plot_cost_penalty_effect(axes[0, 1])

        # Plot 3: Alpha Analysis
        self._plot_alpha_effect(axes[1, 0])

        # Plot 4: Improvement Threshold Analysis
        self._plot_improvement_threshold_effect(axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()
        plt.close()

    def _plot_wealth_trajectories(self, ax, bah_results=None):
        """Plot top 5 wealth trajectories"""
        df_sorted = self.df.sort_values('test_final_wealth', ascending=False)
        top_5 = df_sorted.head(5)

        for idx, (_, row) in enumerate(top_5.iterrows()):
            label = f"CP={row['cost_penalty']:.1f}, α={row['alpha']:.3f}, IT={row['improvement_threshold']:.4f}"
            linestyle = '-' if idx < 3 else '--'
            ax.plot(row['test_daily_wealth'], label=label, linestyle=linestyle, linewidth=2)

        if bah_results is not None:
            ax.plot(bah_results, label='Buy & Hold', linestyle=':', linewidth=2.5, color='black', alpha=0.7)

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
                               trade_costs_dollars=None):
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

        # Plot modified B&H trajectory (matching initial portfolio)
        if modified_bah_results is not None and len(modified_bah_results) > 0:
            ax.plot(modified_bah_results[0], label='Buy & Hold (Best Config Start)',
                    linestyle='-.', linewidth=2.5, color='tab:orange', alpha=0.8)

        # Add vertical lines for trade dates with width proportional to cost
        # Add vertical lines for trade dates (color ∝ transaction cost)
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