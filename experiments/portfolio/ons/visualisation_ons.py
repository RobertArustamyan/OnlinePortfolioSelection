import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ExperimentPlotter:
    def __init__(self, experiment_dir):
        """
        Initialize plotter with experiment directory

        Args:
            experiment_dir: Path to folder containing detailed_results.json
        """
        self.experiment_dir = Path(experiment_dir)

        # Load data
        with open(self.experiment_dir / 'detailed_results.json', 'r') as f:
            self.data = json.load(f)

        with open(self.experiment_dir / 'experiment_metadata.json', 'r') as f:
            self.metadata = json.load(f)

        # Extract model name (should be only one)
        self.model_name = list(self.data.keys())[0]
        self.model_data = self.data[self.model_name]

        # Stock names
        self.stocks = self.metadata['experiment_settings']['stocks']
        self.n_stocks = len(self.stocks)

        print(f"Loaded experiment: {self.model_name}")
        print(f"Stocks: {', '.join(self.stocks)}")
        print(f"Total hyperparameter configurations: {len(self.model_data['all_parameter_combinations'])}")
        print(f"Best parameters: {self.model_data['best_parameters']}\n")

    def print_report_statistics(self):
        """
        Print all statistics needed for the LaTeX report
        """
        print(f"\n{'=' * 80}")
        print(f"REPORT STATISTICS FOR LATEX")
        print(f"{'=' * 80}\n")

        # Get best config
        best_params = self.model_data['best_parameters']

        # Find the matching config
        best_config = None
        for config in self.model_data['all_parameter_combinations']:
            if config['parameters'] == best_params:
                best_config = config
                break

        test_data = self.model_data['test']

        print(f"EXPERIMENT: {', '.join(self.stocks)}")
        print(f"\n--- BEST HYPERPARAMETERS (from validation) ---")
        print(f"Delta (δ):          {best_params['delta']}")
        print(f"Cost Penalty (λ):   {best_params['cost_penalty']}")
        print(f"Alpha (α):          {best_params['alpha']}")

        if best_config:
            print(f"\n--- VALIDATION PERFORMANCE ---")
            print(f"Validation Final Wealth: {best_config['validation']['final_wealth']:.4f}")
            print(f"Validation Return:       {(best_config['validation']['final_wealth'] - 1.0) * 100:+.2f}%")

        print(f"\n--- TEST PERFORMANCE ---")
        final_wealth = test_data['final_wealth']
        net_return_pct = test_data['net_return_pct']
        print(f"Test Final Wealth:   {final_wealth:.4f}")
        print(f"Test Return:         {net_return_pct:+.2f}%")
        print(f"Trade Frequency:     {test_data['trade_frequency']:.2%}")

        print(f"\n--- TRANSACTION COSTS ---")
        costs_dollars = np.array(test_data['transaction_costs_dollars'])
        traded_flags = np.array(test_data['traded_flags'])
        num_trades = np.sum(traded_flags)
        total_cost_usd = np.sum(costs_dollars)
        avg_cost_usd = np.mean([c for c in costs_dollars if c > 0]) if any(costs_dollars) else 0
        cost_drag_pct = test_data['cost_drag_pct']

        print(f"Total Cost Drag:     {cost_drag_pct:.4f}%")
        print(f"Total Cost (USD):    ${total_cost_usd:.2f}")
        print(f"Number of Trades:    {num_trades}")
        print(f"Avg Cost per Trade:  ${avg_cost_usd:.2f}")

        print(f"\n--- BENCHMARKS (Test Period) ---")
        benchmarks = self.metadata['benchmarks']

        if 'buy_and_hold_uniform' in benchmarks:
            bah_uniform = benchmarks['buy_and_hold_uniform']['final_wealth']
            bah_uniform_return = (bah_uniform - 1.0) * 100
            print(f"BAH (Uniform):       {bah_uniform:.4f} ({bah_uniform_return:+.2f}%)")

        if 'buy_and_hold_ons_initial' in benchmarks:
            bah_ons_data = benchmarks['buy_and_hold_ons_initial'].get(self.model_name)
            if bah_ons_data:
                bah_ons = bah_ons_data['final_wealth']
                bah_ons_return = (bah_ons - 1.0) * 100
                print(f"BAH (ONS Initial):   {bah_ons:.4f} ({bah_ons_return:+.2f}%)")

        if benchmarks.get('nasdaq', {}).get('available'):
            nasdaq = benchmarks['nasdaq']['final_wealth']
            nasdaq_return = (nasdaq - 1.0) * 100
            print(f"NASDAQ:              {nasdaq:.4f} ({nasdaq_return:+.2f}%)")

        if benchmarks.get('sp500', {}).get('available'):
            sp500 = benchmarks['sp500']['final_wealth']
            sp500_return = (sp500 - 1.0) * 100
            print(f"S&P 500:             {sp500:.4f} ({sp500_return:+.2f}%)")

        print(f"\n--- LATEX COPY-PASTE VALUES ---")
        print(f"Best delta:                 {best_params['delta']}")
        print(f"Best lambda:                {best_params['cost_penalty']}")
        print(f"Best alpha:                 {best_params['alpha']}")
        if best_config:
            print(f"Validation final wealth:    {best_config['validation']['final_wealth']:.4f}")
        print(f"Test final wealth:          {final_wealth:.4f}")
        print(f"Test return:                {net_return_pct:.2f}\\%")
        print(f"Cost drag:                  {cost_drag_pct:.4f}\\%")
        print(f"Total cost USD:             \\${total_cost_usd:.2f}")
        print(f"Number of trades:           {num_trades}")
        print(f"Avg cost per trade:         \\${avg_cost_usd:.2f}")
        print(f"Trade frequency:            {test_data['trade_frequency']:.2%}")

        print(f"\n{'=' * 80}\n")

    def plot_test_wealth_and_allocation(self, save_path=None):
        """
        Plot 1: Test wealth over time + portfolio allocation
        Left: Wealth evolution with all benchmarks
        Right: Individual portfolio weights (n subplots, one per stock)
        """
        # Create figure with 1 column for wealth + n columns for portfolio allocation
        fig = plt.figure(figsize=(16, max(6, self.n_stocks * 2.5)))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        gs_right = gs[1].subgridspec(self.n_stocks, 1, hspace=0.4)

        # ========== LEFT: Wealth with benchmarks ==========
        ax_wealth = fig.add_subplot(gs[0, 0])

        test_data = self.model_data['test']
        daily_wealth = np.array(test_data['daily_wealth'])
        days = np.arange(len(daily_wealth))

        # ONS wealth
        ax_wealth.plot(days, daily_wealth, linewidth=2.5, color='#2E4053',
                       label='ONS (Cost-Aware)', linestyle='-', zorder=5)

        # Benchmarks
        benchmarks = self.metadata['benchmarks']

        # Uniform BAH (solid line)
        if 'buy_and_hold_uniform' in benchmarks:
            bah_uniform_wealth = np.array(benchmarks['buy_and_hold_uniform']['daily_wealth'])
            ax_wealth.plot(days, bah_uniform_wealth, linewidth=2, color='#27AE60',
                           label='BAH (Uniform)', linestyle='-', alpha=0.8)

        # ONS Initial Portfolio BAH (dashed line)
        if 'buy_and_hold_ons_initial' in benchmarks:
            bah_ons_data = benchmarks['buy_and_hold_ons_initial'].get(self.model_name)
            if bah_ons_data:
                bah_ons_wealth = np.array(bah_ons_data['daily_wealth'])
                ax_wealth.plot(days, bah_ons_wealth, linewidth=2, color='#8E44AD',
                               label='BAH (ONS Initial)', linestyle='--', alpha=0.8)

        # NASDAQ (solid line)
        if benchmarks.get('nasdaq', {}).get('available'):
            nasdaq_wealth = np.array(benchmarks['nasdaq']['daily_wealth'])
            ax_wealth.plot(days, nasdaq_wealth, linewidth=2, color='#E74C3C',
                           label='NASDAQ', linestyle='-', alpha=0.8)

        # S&P 500 (solid line)
        if benchmarks.get('sp500', {}).get('available'):
            sp500_wealth = np.array(benchmarks['sp500']['daily_wealth'])
            ax_wealth.plot(days, sp500_wealth, linewidth=2, color='#3498DB',
                           label='S&P 500', linestyle='-', alpha=0.8)

        ax_wealth.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax_wealth.set_ylabel('Wealth (Normalized)', fontsize=12, fontweight='bold')
        ax_wealth.set_title('Test Period: Wealth Evolution', fontsize=14, fontweight='bold')
        ax_wealth.legend(fontsize=9, frameon=True, shadow=True, loc='best')
        ax_wealth.grid(True, alpha=0.3, linestyle='--')

        # Add final wealth annotation in upper right
        final_wealth = daily_wealth[-1]
        net_return = (final_wealth - 1.0) * 100
        ax_wealth.text(0.98, 0.98, f'ONS Final: {final_wealth:.3f}\n({net_return:+.2f}%)',
                       transform=ax_wealth.transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue',
                                 edgecolor='black', alpha=0.8, linewidth=1.5))

        # ========== RIGHT: Portfolio allocation (one subplot per stock, stacked vertically) ==========
        portfolios = np.array(test_data['portfolios_used'])  # Shape: (num_days, n_stocks)
        portfolio_days = np.arange(1, len(portfolios) + 1)

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
                  '#1ABC9C', '#E67E22', '#34495E', '#95A5A6', '#D35400']

        for i in range(self.n_stocks):
            ax = fig.add_subplot(gs_right[i, 0])

            ax.plot(portfolio_days, portfolios[:, i],
                    linewidth=2, color=colors[i % len(colors)], alpha=0.85)

            ax.set_title(self.stocks[i], fontsize=10, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.set_ylabel('Weight', fontsize=8)

            # Only show x-label on last subplot
            if i == self.n_stocks - 1:
                ax.set_xlabel('Day', fontsize=9)
            else:
                ax.set_xticklabels([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def plot_transaction_costs(self, save_path=None):
        """
        Plot 4: Transaction costs over time
        Top: Cost as % of wealth + Cost in USD (2 y-axes)
        Bottom: Cumulative cost
        """
        fig, (ax1_top, ax1_bot) = plt.subplots(2, 1, figsize=(14, 10))

        test_data = self.model_data['test']
        costs = np.array(test_data['transaction_costs'])
        costs_dollars = np.array(test_data['transaction_costs_dollars'])
        days = np.arange(len(costs))

        # ========== TOP SUBPLOT: Cost % and Cost USD ==========
        # First y-axis: Cost as % of wealth
        color1 = '#E74C3C'
        ax1_top.plot(days, costs * 100, linewidth=2, color=color1, label='Cost (% of wealth)', alpha=0.8)
        ax1_top.set_ylabel('Cost (% of wealth)', fontsize=12, fontweight='bold', color=color1)
        ax1_top.tick_params(axis='y', labelcolor=color1)
        ax1_top.grid(True, alpha=0.3, linestyle='--')
        ax1_top.set_title('Transaction Costs Over Time', fontsize=14, fontweight='bold')

        # Second y-axis: Cost in dollars
        ax2_top = ax1_top.twinx()
        color2 = '#3498DB'
        ax2_top.bar(days, costs_dollars, width=1.0, alpha=0.4, color=color2, label='Cost (USD)')
        ax2_top.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold', color=color2)
        ax2_top.tick_params(axis='y', labelcolor=color2)

        # Combine legends
        lines1, labels1 = ax1_top.get_legend_handles_labels()
        lines2, labels2 = ax2_top.get_legend_handles_labels()
        ax1_top.legend(lines1 + lines2, labels1 + labels2,
                       loc='upper left', fontsize=10, frameon=True, shadow=True)

        # ========== BOTTOM SUBPLOT: Cumulative Cost ==========
        color3 = '#2ECC71'
        cumulative_cost = np.cumsum(costs) * 100
        ax1_bot.plot(days, cumulative_cost, linewidth=2.5, color=color3,
                     linestyle='-', alpha=0.9, label='Cumulative Cost')
        ax1_bot.fill_between(days, 0, cumulative_cost, alpha=0.2, color=color3)

        ax1_bot.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax1_bot.set_ylabel('Cumulative Cost (% of initial wealth)', fontsize=12, fontweight='bold', color=color3)
        ax1_bot.tick_params(axis='y', labelcolor=color3)
        ax1_bot.set_title('Cumulative Transaction Costs', fontsize=14, fontweight='bold')
        ax1_bot.grid(True, alpha=0.3, linestyle='--')
        ax1_bot.legend(fontsize=10, frameon=True, shadow=True, loc='upper left')

        # Summary statistics
        total_cost_pct = test_data['cost_drag_pct']
        total_cost_usd = np.sum(costs_dollars)
        avg_cost_usd = np.mean([c for c in costs_dollars if c > 0]) if any(costs_dollars) else 0
        num_trades = np.sum(np.array(test_data['traded_flags']))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def plot_all_configs_training_wealth(self, save_path=None):
        """
        Plot 7: All hyperparameter configurations (training wealth evolution)
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        all_configs = self.model_data['all_parameter_combinations']

        # Define color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(all_configs)))

        # Plot each configuration
        for i, config in enumerate(all_configs):
            params = config['parameters']
            train_wealth = np.array(config['training']['daily_wealth'])
            days = np.arange(len(train_wealth))

            label = f"δ={params['delta']}, λ={params['cost_penalty']}, α={params['alpha']}"
            ax.plot(days, train_wealth, alpha=0.65, linewidth=2,
                    color=colors[i], label=label if i < 12 else "")

        ax.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Wealth (Normalized)', fontsize=12, fontweight='bold')
        ax.set_title(f'All Hyperparameter Configurations: Training Wealth Evolution ({len(all_configs)} configs)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='upper left', ncol=2, frameon=True, shadow=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        plt.show()

    def create_all_plots(self, output_dir=None, print_stats=True):
        """
        Create selected plots for report and save to output directory
        """
        if output_dir is None:
            output_dir = self.experiment_dir / 'plots'
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Print statistics first
        if print_stats:
            self.print_report_statistics()

        print(f"\n{'=' * 80}")
        print(f"Creating report plots...")
        print(f"Output directory: {output_dir}")
        print(f"{'=' * 80}\n")

        # Only create plots 1, 4, and 7
        plots = [
            ('test_wealth_and_allocation.png', self.plot_test_wealth_and_allocation),
            ('transaction_costs.png', self.plot_transaction_costs),
            ('all_configs_training.png', self.plot_all_configs_training_wealth),
        ]

        for filename, plot_func in plots:
            print(f"Creating: {filename}")
            save_path = output_dir / filename
            plot_func(save_path=save_path)
            plt.close('all')  # Close to free memory

        print(f"\n{'=' * 80}")
        print(f"✓ All plots created successfully!")
        print(f"Report-ready plots: {len(plots)}")
        print(f"{'=' * 80}\n")
def main():
    experiment_dir = "/home/robert/PycharmProjects/OnlinePortfolioSelection/results/portfolio/WithCosts/ONS/optuna_10_25-02-26_19-39-01"
    plotter = ExperimentPlotter(experiment_dir)

    # Create all plots
    plotter.create_all_plots()


if __name__ == "__main__":
    main()