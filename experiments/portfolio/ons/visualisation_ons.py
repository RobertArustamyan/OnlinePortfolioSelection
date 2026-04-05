import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import pandas as pd

# Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Global grid fix: make grid lines visible on all backends
plt.rcParams['grid.color'] = '#CCCCCC'
plt.rcParams['grid.linewidth'] = 0.6
plt.rcParams['grid.alpha'] = 0.7


class ExperimentPlotter:
    def __init__(self, experiment_dir, initial_capital=None, optimize_metric=None):
        self.experiment_dir = Path(experiment_dir)
        self.initial_capital = initial_capital
        self.optimize_metric = optimize_metric

        with open(self.experiment_dir / 'detailed_results.json', 'r') as f:
            self.data = json.load(f)
        with open(self.experiment_dir / 'experiment_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        with open(self.experiment_dir / 'benchmarks.json', 'r') as f:
            self.benchmarks = json.load(f)

        self.model_name = list(self.data.keys())[0]
        self.model_data = self.data[self.model_name]
        self.stocks = self.metadata['experiment_settings']['stocks']
        self.n_stocks = len(self.stocks)

        # Pre-compute commonly used arrays
        self.test_data = self.model_data['test']
        self.daily_wealth = np.array(self.test_data['daily_wealth'])
        self.portfolios = np.array(self.test_data['portfolios_used'])
        self.price_relatives = np.array(self.test_data['test_price_relatives'])
        self.costs = np.array(self.test_data['transaction_costs'])
        self.costs_dollars = np.array(self.test_data['transaction_costs_dollars'])
        self.turnovers = np.array(self.test_data['turnovers'])
        self.traded_flags = np.array(self.test_data['traded_flags'])
        self.n_test_days = len(self.costs)

        # Pre-compute daily returns
        self.daily_returns = np.diff(self.daily_wealth) / self.daily_wealth[:-1]

        print(f"Loaded: {self.model_name}")
        print(f"Stocks: {', '.join(self.stocks)}")
        print(f"Test days: {self.n_test_days}")
        print(f"Best params: {self.model_data['best_parameters']}\n")

    def _compute_drawdown_series(self, wealth_array):
        """Compute drawdown series from wealth array."""
        cumulative = wealth_array / wealth_array[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        return drawdown

    def _compute_rolling_metric(self, returns, window, metric='sharpe'):
        """Compute rolling Sharpe or Sortino ratio."""
        n = len(returns)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            chunk = returns[i - window + 1:i + 1]
            mean_r = np.mean(chunk)
            if metric == 'sharpe':
                std_r = np.std(chunk)
                result[i] = (mean_r / std_r) * np.sqrt(252) if std_r > 1e-10 else 0.0
            elif metric == 'sortino':
                downside = chunk[chunk < 0]
                ds_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 1 else 1e-10
                result[i] = (mean_r / ds_std) * np.sqrt(252) if ds_std > 1e-10 else 0.0
        return result

    def _grid_kwargs(self):
        """Standard grid settings for all plots."""
        return dict(visible=True, alpha=0.6, linestyle='--', color='#CCCCCC')

    def plot_test_wealth_and_allocation(self, save_path=None):
        """Wealth evolution with benchmarks, portfolio weights, and per-stock cumulative returns."""
        fig_width = 14 + min(self.n_stocks, 10)
        fig_height = max(6, 1.5 * self.n_stocks)
        fig = plt.figure(figsize=(fig_width, fig_height))

        gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])
        gs_weights = gs[1].subgridspec(self.n_stocks, 1, hspace=0.4)
        gs_returns = gs[2].subgridspec(self.n_stocks, 1, hspace=0.4)

        ax_wealth = fig.add_subplot(gs[0, 0])
        days = np.arange(len(self.daily_wealth))

        ons_final = self.daily_wealth[-1]
        ons_ret = (ons_final - 1.0) * 100
        ax_wealth.plot(days, self.daily_wealth, linewidth=2.5, color='#2E4053',
                       label=f'ONS (Cost-Aware): {ons_final:.3f}',
                       linestyle='-', zorder=5)

        benchmark_cfg = [
            ('uniform', '#27AE60', 'BAH (Uniform)', '-'),
            ('ons_initial', '#8E44AD', 'BAH (ONS Initial)', '--'),
            ('nasdaq', '#E74C3C', 'NASDAQ', '-'),
            ('sp500', '#3498DB', 'S&P 500', '-'),
        ]
        for bname, bcolor, blabel, bstyle in benchmark_cfg:
            if bname in self.benchmarks:
                bw = np.array(self.benchmarks[bname]['daily_wealth'])
                bfinal = bw[-1]
                bret = (bfinal - 1.0) * 100
                ax_wealth.plot(days[:len(bw)], bw, linewidth=2, color=bcolor,
                               label=f'{blabel}: {bfinal:.3f}',
                               linestyle=bstyle, alpha=0.8)

        regime_data = self.model_data.get('regime_analysis', {})
        test_regime = regime_data.get('test', {}).get('daily_regime', [])
        self._shade_regimes(ax_wealth, test_regime)

        ax_wealth.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax_wealth.set_ylabel('Wealth (Normalized)', fontsize=12, fontweight='bold')
        # Build title with optional initial_capital and optimize_metric
        title_parts = ['Test Period: Wealth Evolution']
        if self.initial_capital is not None:
            title_parts.append(f'${self.initial_capital:,}')
        if self.optimize_metric is not None:
            title_parts.append(f'Optimized: {self.optimize_metric}')
        ax_wealth.set_title('  |  '.join(title_parts), fontsize=14, fontweight='bold')
        ax_wealth.legend(fontsize=9, frameon=True, shadow=True, loc='best')
        ax_wealth.grid(**self._grid_kwargs())

        portfolio_days = np.arange(1, len(self.portfolios) + 1)
        cumulative_returns = np.cumprod(self.price_relatives, axis=0)

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
                  '#1ABC9C', '#E67E22', '#34495E', '#95A5A6', '#D35400']

        for i in range(self.n_stocks):
            color = colors[i % len(colors)]

            ax_w = fig.add_subplot(gs_weights[i])
            ax_w.plot(portfolio_days, self.portfolios[:, i], linewidth=2, color=color, alpha=0.85)
            ax_w.set_title(self.stocks[i], fontsize=10, fontweight='bold')
            ax_w.set_ylim([0, 1])
            ax_w.grid(**self._grid_kwargs())
            ax_w.set_ylabel('Weight', fontsize=8)
            if i < self.n_stocks - 1:
                ax_w.set_xticklabels([])
            else:
                ax_w.set_xlabel('Day', fontsize=9)

            ax_r = fig.add_subplot(gs_returns[i])
            ax_r.plot(portfolio_days, cumulative_returns[:, i], linewidth=2, color=color, alpha=0.85)
            ax_r.set_title(self.stocks[i], fontsize=10, fontweight='bold')
            ax_r.grid(**self._grid_kwargs())
            ax_r.set_ylabel('Cum. Ret.', fontsize=8)
            if i < self.n_stocks - 1:
                ax_r.set_xticklabels([])
            else:
                ax_r.set_xlabel('Day', fontsize=9)

        fig.text(0.68, 1.01, 'Portfolio Weight', ha='center', fontsize=11, fontweight='bold',
                 transform=fig.transFigure)
        fig.text(0.88, 1.01, 'Cumulative Return', ha='center', fontsize=11, fontweight='bold',
                 transform=fig.transFigure)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.show()

    def plot_drawdown_and_risk(self, rolling_window=60, save_path=None):
        """
        Top: Drawdown over time (ONS vs benchmarks)
        Bottom: Rolling Sortino ratio
        """
        fig, (ax_dd, ax_sortino) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[1, 1])

        days = np.arange(len(self.daily_wealth))
        days_ret = np.arange(len(self.daily_returns))

        # ===== TOP: Drawdown comparison =====
        dd_ons = self._compute_drawdown_series(self.daily_wealth)
        ax_dd.fill_between(days, 0, -dd_ons * 100, alpha=0.4, color='#2E4053', label='ONS')
        ax_dd.plot(days, -dd_ons * 100, linewidth=1.5, color='#2E4053')

        for bname, bcolor, bstyle in [
            ('uniform', '#27AE60', '-'), ('ons_initial', '#8E44AD', '--'),
            ('nasdaq', '#E74C3C', '-'), ('sp500', '#3498DB', '-')
        ]:
            if bname in self.benchmarks:
                bw = np.array(self.benchmarks[bname]['daily_wealth'])
                bdd = self._compute_drawdown_series(bw)
                label = {'uniform': 'BAH (Uniform)', 'ons_initial': 'BAH (ONS Init)',
                         'nasdaq': 'NASDAQ', 'sp500': 'S&P 500'}[bname]
                ax_dd.plot(days[:len(bdd)], -bdd * 100, linewidth=1.5, color=bcolor,
                           linestyle=bstyle, alpha=0.7, label=label)

        max_dd = np.max(dd_ons) * 100
        max_dd_day = np.argmax(dd_ons)
        ax_dd.axvline(x=max_dd_day, color='red', linestyle=':', alpha=0.5)
        ax_dd.annotate(f'Max DD: {max_dd:.1f}% (day {max_dd_day})',
                       xy=(max_dd_day, -max_dd), fontsize=9, color='red',
                       xytext=(max_dd_day + 20, -max_dd - 2),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1))

        ax_dd.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax_dd.set_title('Underwater Plot (Drawdown from Peak)', fontsize=14, fontweight='bold')
        ax_dd.legend(fontsize=9, frameon=True, loc='lower left', ncol=3)
        ax_dd.grid(**self._grid_kwargs())

        rolling_sortino = self._compute_rolling_metric(self.daily_returns, rolling_window, 'sortino')
        ax_sortino.plot(days_ret, rolling_sortino, linewidth=2, color='#2E4053', label=f'ONS ({rolling_window}d)')

        # Benchmarks rolling sortino
        for bname, bcolor in [('uniform', '#27AE60'), ('nasdaq', '#E74C3C')]:
            if bname in self.benchmarks:
                bw = np.array(self.benchmarks[bname]['daily_wealth'])
                br = np.diff(bw) / bw[:-1]
                b_sortino = self._compute_rolling_metric(br, rolling_window, 'sortino')
                label = 'BAH (Uniform)' if bname == 'uniform' else 'NASDAQ'
                ax_sortino.plot(np.arange(len(br)), b_sortino, linewidth=1.5, color=bcolor,
                                alpha=0.7, label=label)

        ax_sortino.axhline(y=0, color='red', linestyle='-', alpha=0.4, linewidth=1)
        ax_sortino.axhline(y=1, color='green', linestyle=':', alpha=0.4, linewidth=1)
        ax_sortino.axhline(y=2, color='green', linestyle=':', alpha=0.3, linewidth=1)

        ax_sortino.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax_sortino.set_ylabel('Rolling Sortino', fontsize=12, fontweight='bold')
        ax_sortino.set_title(f'Rolling Sortino Ratio ({rolling_window}-day window)', fontsize=14, fontweight='bold')
        ax_sortino.legend(fontsize=9, frameon=True, loc='best')
        ax_sortino.grid(**self._grid_kwargs())

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.show()


    def plot_transaction_costs(self, save_path=None):
        """
        Top-left: Per-trade cost (% wealth) + USD on twin axis
        Top-right: Cumulative cost drag over time
        Bottom-left: Turnover per day
        Bottom-right: Cost efficiency (cost per unit of turnover)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        days = np.arange(self.n_test_days)

        ax = axes[0, 0]
        color1, color2 = '#E74C3C', '#3498DB'
        ax.plot(days, self.costs * 100, linewidth=1.5, color=color1, alpha=0.8, label='Cost (% wealth)')
        ax.set_ylabel('Cost (% of wealth)', fontsize=10, fontweight='bold', color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(**self._grid_kwargs())
        ax.set_title('Per-Trade Transaction Costs', fontsize=12, fontweight='bold')

        ax2 = ax.twinx()
        ax2.bar(days, self.costs_dollars, width=1.0, alpha=0.3, color=color2, label='Cost (USD)')
        ax2.set_ylabel('Cost (USD)', fontsize=10, fontweight='bold', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

        ax = axes[0, 1]
        cumulative_cost_pct = np.cumsum(self.costs) * 100
        cumulative_cost_usd = np.cumsum(self.costs_dollars)
        ax.plot(days, cumulative_cost_pct, linewidth=2.5, color='#2ECC71', label='Cumulative Cost (%)')
        ax.fill_between(days, 0, cumulative_cost_pct, alpha=0.15, color='#2ECC71')
        ax.set_ylabel('Cumulative Cost (% of wealth)', fontsize=10, fontweight='bold', color='#2ECC71')
        ax.set_title('Cumulative Transaction Cost Drag', fontsize=12, fontweight='bold')
        ax.grid(**self._grid_kwargs())

        ax2 = ax.twinx()
        ax2.plot(days, cumulative_cost_usd, linewidth=1.5, color='#E67E22', linestyle='--', alpha=0.8)
        ax2.set_ylabel('Cumulative Cost (USD)', fontsize=10, fontweight='bold', color='#E67E22')

        total_usd = cumulative_cost_usd[-1]
        ax.text(0.02, 0.95, f'Total: {cumulative_cost_pct[-1]:.3f}%  (${total_usd:.2f})',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax = axes[1, 0]
        ax.bar(days, self.turnovers, width=1.0, alpha=0.6, color='#9B59B6')
        ax.set_xlabel('Trading Day', fontsize=10, fontweight='bold')
        ax.set_ylabel('Turnover (L1 distance)', fontsize=10, fontweight='bold')
        ax.set_title('Daily Portfolio Turnover', fontsize=12, fontweight='bold')
        ax.grid(**self._grid_kwargs())

        # Annotate trade frequency
        n_trades = np.sum(self.traded_flags)
        trade_freq = n_trades / self.n_test_days
        ax.text(0.98, 0.95, f'Trade freq: {trade_freq:.1%}\n({n_trades}/{self.n_test_days} days)',
                transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

        ax = axes[1, 1]
        traded_mask = self.traded_flags.astype(bool)
        if np.sum(traded_mask) > 0:
            ax.scatter(self.turnovers[traded_mask], self.costs_dollars[traded_mask],
                       alpha=0.5, s=20, color='#E74C3C', edgecolors='none')
            ax.set_xlabel('Turnover', fontsize=10, fontweight='bold')
            ax.set_ylabel('Cost (USD)', fontsize=10, fontweight='bold')
            ax.set_title('Cost vs Turnover (traded days only)', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No trades during test period', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')
            ax.set_title('Cost vs Turnover', fontsize=12, fontweight='bold')
        ax.grid(**self._grid_kwargs())

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.show()

    def plot_relative_performance(self, save_path=None):
        """
        Top: ONS wealth / BAH(Uniform) wealth — shows when algorithm adds value
        Bottom: Daily return distribution comparison (ONS vs BAH)
        """
        fig, (ax_rel, ax_dist) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[1.2, 1])

        days = np.arange(len(self.daily_wealth))

        if 'uniform' in self.benchmarks:
            bah_wealth = np.array(self.benchmarks['uniform']['daily_wealth'])
            min_len = min(len(self.daily_wealth), len(bah_wealth))
            relative = self.daily_wealth[:min_len] / bah_wealth[:min_len]

            ax_rel.plot(days[:min_len], relative, linewidth=2, color='#2E4053')
            ax_rel.fill_between(days[:min_len], 1, relative, where=relative >= 1,
                                alpha=0.2, color='green', label='Outperforming BAH')
            ax_rel.fill_between(days[:min_len], 1, relative, where=relative < 1,
                                alpha=0.2, color='red', label='Underperforming BAH')
            ax_rel.axhline(y=1, color='black', linestyle='-', linewidth=1, alpha=0.5)

            # Percentage of days outperforming
            pct_outperform = np.sum(relative > 1) / len(relative) * 100
            ax_rel.text(0.02, 0.95, f'Outperforming: {pct_outperform:.1f}% of days',
                        transform=ax_rel.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax_rel.set_ylabel('ONS / BAH Wealth Ratio', fontsize=12, fontweight='bold')
        ax_rel.set_title('Relative Performance vs Buy-and-Hold (Uniform)', fontsize=14, fontweight='bold')
        ax_rel.legend(fontsize=9, frameon=True, loc='lower left')
        ax_rel.grid(**self._grid_kwargs())

        ons_returns = self.daily_returns * 100  # to percentage

        ax_dist.hist(ons_returns, bins=60, alpha=0.6, color='#2E4053', density=True,
                     label=f'ONS (mean={np.mean(ons_returns):.3f}%, std={np.std(ons_returns):.3f}%)')

        if 'uniform' in self.benchmarks:
            bah_w = np.array(self.benchmarks['uniform']['daily_wealth'])
            bah_returns = np.diff(bah_w) / bah_w[:-1] * 100
            ax_dist.hist(bah_returns, bins=60, alpha=0.4, color='#27AE60', density=True,
                         label=f'BAH (mean={np.mean(bah_returns):.3f}%, std={np.std(bah_returns):.3f}%)')

        ax_dist.axvline(x=0, color='red', linestyle='-', alpha=0.4, linewidth=1)
        ax_dist.set_xlabel('Daily Return (%)', fontsize=12, fontweight='bold')
        ax_dist.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax_dist.set_title('Daily Return Distribution', fontsize=14, fontweight='bold')
        ax_dist.legend(fontsize=9, frameon=True, loc='upper left')
        ax_dist.grid(**self._grid_kwargs())

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.show()


    def plot_portfolio_concentration(self, save_path=None):
        """
        Top: Stacked area chart of portfolio weights (full picture at once)
        Bottom: HHI concentration index + cash weight over time
        """
        fig, (ax_stack, ax_hhi) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[1.2, 1])

        days = np.arange(1, len(self.portfolios) + 1)

        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
                  '#1ABC9C', '#E67E22', '#34495E', '#95A5A6', '#D35400']

        ax_stack.stackplot(days, self.portfolios.T,
                           labels=self.stocks, colors=colors[:self.n_stocks], alpha=0.8)
        ax_stack.set_ylabel('Portfolio Weight', fontsize=12, fontweight='bold')
        ax_stack.set_title('Portfolio Allocation Over Time', fontsize=14, fontweight='bold')
        ax_stack.set_ylim([0, 1])
        ax_stack.legend(fontsize=8, loc='upper left', ncol=min(self.n_stocks, 6), frameon=True)
        ax_stack.grid(**self._grid_kwargs())

        # HHI = sum of squared weights. 1/n = perfectly diversified, 1.0 = fully concentrated
        hhi = np.sum(self.portfolios ** 2, axis=1)
        min_hhi = 1.0 / self.n_stocks  # theoretical minimum (uniform)

        color_hhi = '#2E4053'
        ax_hhi.plot(days, hhi, linewidth=2, color=color_hhi, label='HHI (concentration)')
        ax_hhi.axhline(y=min_hhi, color=color_hhi, linestyle=':', alpha=0.5,
                        label=f'Uniform HHI = {min_hhi:.3f}')
        ax_hhi.set_ylabel('HHI (Concentration)', fontsize=12, fontweight='bold', color=color_hhi)
        ax_hhi.tick_params(axis='y', labelcolor=color_hhi)
        ax_hhi.set_ylim([0, 1.05])

        # Cash weight on secondary axis (if cash exists)
        # Detect cash: check if any stock name contains 'cash' (case-insensitive)
        cash_idx = None
        for i, s in enumerate(self.stocks):
            if 'cash' in s.lower():
                cash_idx = i
                break

        if cash_idx is not None:
            ax2 = ax_hhi.twinx()
            cash_weights = self.portfolios[:, cash_idx]
            ax2.fill_between(days, 0, cash_weights * 100, alpha=0.15, color='#27AE60')
            ax2.plot(days, cash_weights * 100, linewidth=1.5, color='#27AE60',
                     alpha=0.8, label='Cash weight')
            ax2.set_ylabel('Cash Weight (%)', fontsize=12, fontweight='bold', color='#27AE60')
            ax2.tick_params(axis='y', labelcolor='#27AE60')
            ax2.set_ylim([0, 105])
            ax2.legend(fontsize=9, loc='upper right', frameon=True)

        ax_hhi.set_xlabel('Trading Day', fontsize=12, fontweight='bold')
        ax_hhi.set_title('Portfolio Concentration (HHI) & Cash Allocation', fontsize=14, fontweight='bold')
        ax_hhi.legend(fontsize=9, loc='upper left', frameon=True)
        ax_hhi.grid(**self._grid_kwargs())

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        plt.show()

    def print_report_statistics(self):
        best_params = self.model_data['best_parameters']
        test_data = self.test_data

        best_config = None
        for config in self.model_data['all_parameter_combinations']:
            if config['parameters'] == best_params:
                best_config = config
                break

        # Compute risk metrics
        dd = self._compute_drawdown_series(self.daily_wealth)
        max_dd = np.max(dd)
        n_days = len(self.daily_returns)
        ann_return = (self.daily_wealth[-1] / self.daily_wealth[0]) ** (252 / n_days) - 1
        std_r = np.std(self.daily_returns)
        sharpe = (np.mean(self.daily_returns) / std_r) * np.sqrt(252) if std_r > 1e-10 else 0
        downside = self.daily_returns[self.daily_returns < 0]
        ds_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 1 else 1e-10
        sortino = (np.mean(self.daily_returns) / ds_std) * np.sqrt(252)
        calmar = ann_return / max_dd if max_dd > 1e-10 else 0

        costs_dollars = np.array(test_data['transaction_costs_dollars'])
        num_trades = np.sum(np.array(test_data['traded_flags']))
        total_cost_usd = np.sum(costs_dollars)
        avg_cost_usd = np.mean([c for c in costs_dollars if c > 0]) if any(costs_dollars) else 0

        print(f"\nExperiment: {', '.join(self.stocks)}")
        print(f"\nBest Hyperparameters")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        if best_config:
            print(f"\nValidation")
            print(f"  Final Wealth: {best_config['validation']['final_wealth']:.4f}")

        print(f"\nTest Performance")
        print(f"Final Wealth:    {test_data['final_wealth']:.4f}")
        print(f"Net Return:      {test_data['net_return_pct']:+.2f}%")
        print(f"Ann. Return:     {ann_return*100:+.2f}%")
        print(f"Max Drawdown:    {max_dd*100:.2f}%")
        print(f"Sharpe Ratio:    {sharpe:.4f}")
        print(f"Sortino Ratio:   {sortino:.4f}")
        print(f"Calmar Ratio:    {calmar:.4f}")
        print(f"Trade Frequency: {test_data['trade_frequency']:.2%}")

        print(f"\nTransaction Costs")
        print(f"Cost Drag:       {test_data['cost_drag_pct']:.4f}%")
        print(f"Total Cost:      ${total_cost_usd:.2f}")
        print(f"Num Trades:      {num_trades}")
        print(f"Avg Cost/Trade:  ${avg_cost_usd:.2f}")

        print(f"\nBenchmarks (Test)")
        for name, result in self.benchmarks.items():
            fw = result['final_wealth']
            print(f"  {name:20s}: {fw:.4f} ({(fw-1)*100:+.2f}%)")
        print(f"{'='*80}\n")


    def create_all_plots(self, output_dir=None, print_stats=True):
        if output_dir is None:
            output_dir = self.experiment_dir / 'plots'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if print_stats:
            self.print_report_statistics()

        print(f"Creating plots in: {output_dir}\n")

        plots = [
            ('1_wealth_and_allocation.png', self.plot_test_wealth_and_allocation),
            ('2_transaction_costs.png', self.plot_transaction_costs),
            ('3_relative_performance.png', self.plot_relative_performance),
        ]

        for filename, plot_func in plots:
            print(f"  Creating: {filename}")
            plot_func(save_path=output_dir / filename)
            plt.close('all')

        print(f"\nAll {len(plots)} plots created.")

    def _shade_regimes(self, ax, daily_regime):
        """
        Shade background of ax with subtle green (bull) / red (bear) regions.
        daily_regime: list of bool, True=bull, False=bear, aligned with trading days.
        """
        if not daily_regime:
            return

        regime = np.array(daily_regime)
        n = len(regime)
        i = 0
        while i < n:
            val = regime[i]
            j = i
            while j < n and regime[j] == val:
                j += 1
            color = '#b7e4b7' if val else '#f5b7b1'  # muted green / muted red
            ax.axvspan(i, j, alpha=0.25, color=color, linewidth=0)
            i = j

def main():
    experiment_dir = "/home/robert/PycharmProjects/OnlinePortfolioSelection/results/portfolio/WithCosts/ONS_Synthetic/optuna_5_11-03-26_21-50"
    plotter = ExperimentPlotter(experiment_dir)
    plotter.create_all_plots()


if __name__ == "__main__":
    main()