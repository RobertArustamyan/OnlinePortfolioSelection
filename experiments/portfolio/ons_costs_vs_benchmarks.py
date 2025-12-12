from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts
from algorithms.transactions.linear_cost import LinearCost
from algorithms.transactions.fixed_cost_per_asset import FixedCostPerAsset
from algorithms.transactions.fixed_cost_per_rebalancing import FixedCostPerRebalancing
from algorithms.transactions.quadratic_cost import QuadraticCost
from utils.compare_strategies import compare_strategies
from utils.data_prep import prepare_stock_data
from benchmarks.portfolio.buy_and_hold import run_buy_and_hold


class CostModelExperiment:
    def __init__(self, cost_model, model_name, stocks, train_data, test_data):
        self.cost_model = cost_model
        self.model_name = model_name
        self.stocks = stocks
        self.train_data = train_data
        self.test_data = test_data
        self.n_stocks = len(stocks)

        # Compute ONS parameters
        T = len(train_data) + len(test_data)
        self.eta = (self.n_stocks ** 1.25) / (np.sqrt(T * np.log(self.n_stocks * T)))
        self.beta = 1.0 / (8 * (self.n_stocks ** 0.25) * np.sqrt(T * np.log(self.n_stocks * T)))
        self.delta = 0.8

        self.results = []

    def run_single_experiment(self, cost_penalty, alpha, improvement_threshold,
                              continue_from_training=False):
        """Run a single experiment with given parameters"""

        # Initialize algorithm
        ons = OnlineNewtonStepCosts(
            n_stocks=self.n_stocks,
            eta=self.eta,
            beta=self.beta,
            delta=self.delta,
            cost_model=self.cost_model,
            cost_penalty=cost_penalty,
            alpha=alpha,
            improvement_threshold=improvement_threshold
        )

        # Training phase
        train_results = ons.simulate_trading(self.train_data, verbose=False)

        # Testing phase
        if continue_from_training:
            test_daily_wealth = [train_results['daily_wealth'][-1]]
            test_wealth_current = train_results['daily_wealth'][-1]
        else:
            test_daily_wealth = [1.0]
            test_wealth_current = 1.0

        test_costs = []
        test_turnovers = []
        test_no_trades = 0

        for price_rel in self.test_data:
            current_portfolio = ons.get_portfolio()
            prev_portfolio = ons.prev_portfolio.copy()

            # Check if traded
            if np.allclose(current_portfolio, prev_portfolio):
                test_no_trades += 1

            # Compute cost and apply to wealth
            tc = self.cost_model.compute_cost(prev_portfolio, current_portfolio)
            test_wealth_current *= (1 - tc)
            test_costs.append(tc)

            # Compute turnover
            turnover = np.sum(np.abs(current_portfolio - prev_portfolio))
            test_turnovers.append(turnover)

            # Apply returns
            daily_return = np.dot(current_portfolio, price_rel)
            test_wealth_current *= daily_return
            test_daily_wealth.append(test_wealth_current)

            # Update
            ons.update(price_rel, current_portfolio)

        # Compile results
        result = {
            'cost_penalty': cost_penalty,
            'alpha': alpha,
            'improvement_threshold': improvement_threshold,
            'train_final_wealth': train_results['final_wealth'],
            'train_total_cost': train_results['total_transaction_cost'],
            'train_avg_turnover': train_results['avg_turnover'],
            'train_trade_frequency': train_results['trade_frequency'],
            'test_final_wealth': test_wealth_current,
            'test_total_cost': sum(test_costs),
            'test_avg_cost_per_trade': np.mean([c for c in test_costs if c > 0]) if any(test_costs) else 0,
            'test_avg_turnover': np.mean(test_turnovers),
            'test_trade_frequency': 1 - (test_no_trades / len(self.test_data)),
            'test_num_no_trades': test_no_trades,
            'test_daily_wealth': np.array(test_daily_wealth),
            'test_net_return': (test_wealth_current - 1.0) * 100,
            'test_cost_drag': train_results['total_transaction_cost'] * 100
        }

        return result

    def grid_search(self, cost_penalties, alphas, improvement_thresholds,
                    continue_from_training=False, verbose=True, n_jobs=-1):
        """
        Run grid search with parallel execution.

        Args:
            n_jobs: Number of parallel jobs. -1 uses all available cores.
        """
        if verbose:
            print(f"Grid Search for {self.model_name}")
            print(f"Testing {len(cost_penalties)} × {len(alphas)} × {len(improvement_thresholds)} "
                  f"= {len(cost_penalties) * len(alphas) * len(improvement_thresholds)} combinations")
            print(f"Running with n_jobs={n_jobs if n_jobs != -1 else 'all cores'}\n")

        # Create all parameter combinations
        param_combinations = [
            (cp, a, it)
            for cp in cost_penalties
            for a in alphas
            for it in improvement_thresholds
        ]

        # Run experiments in parallel
        self.results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(self.run_single_experiment)(
                cost_penalty=cp,
                alpha=a,
                improvement_threshold=it,
                continue_from_training=continue_from_training
            )
            for cp, a, it in param_combinations
        )

        if verbose:
            print(f"\nCompleted {len(self.results)} experiments")
            # Show best result
            best_result = max(self.results, key=lambda x: x['test_final_wealth'])
            print(f"\nBest result:")
            print(f"  Cost Penalty: {best_result['cost_penalty']}")
            print(f"  Alpha: {best_result['alpha']}")
            print(f"  Improvement Threshold: {best_result['improvement_threshold']}")
            print(f"  Test Final Wealth: {best_result['test_final_wealth']:.4f}")
            print(f"  Trade Frequency: {best_result['test_trade_frequency']:.2%}\n")

        return self.results

    def get_results_dataframe(self):
        """Convert results to pandas DataFrame"""
        df = pd.DataFrame(self.results)
        df['model'] = self.model_name
        return df

    def plot_results(self, save_path=None, continue_from_training=False, bah_results=None):
        """
        Create 4 plots for the cost model experiment:
        1. Top 5 wealth curves + Buy & Hold
        2. Cost penalty vs (Wealth + Trade Frequency)
        3. Alpha vs (Wealth + Trade Frequency)
        4. Improvement threshold vs (Wealth + Trade Frequency)
        """
        df = self.get_results_dataframe()

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.model_name} - Performance Analysis', fontsize=16, fontweight='bold')

        # ========== PLOT 1: Top 5 Wealth Curves + B&H ==========
        ax1 = axes[0, 0]

        # Sort by test final wealth and get top 5
        df_sorted = df.sort_values('test_final_wealth', ascending=False)
        top_5 = df_sorted.head(5)

        # Plot top 5 wealth curves
        for idx, (_, row) in enumerate(top_5.iterrows()):
            label = f"CP={row['cost_penalty']:.1f}, α={row['alpha']:.3f}, IT={row['improvement_threshold']:.4f}"
            linestyle = '-' if idx < 3 else '--'  # Solid for top 3, dashed for 4-5
            ax1.plot(row['test_daily_wealth'], label=label, linestyle=linestyle, linewidth=2)

        # Plot Buy & Hold if provided
        if bah_results is not None:
            ax1.plot(bah_results, label='Buy & Hold', linestyle=':', linewidth=2.5, color='black', alpha=0.7)

        ax1.set_xlabel('Trading Days', fontsize=11)
        ax1.set_ylabel('Wealth', fontsize=11)
        ax1.set_title('Top 5 Wealth Trajectories', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)

        # ========== PLOT 2: Cost Penalty Analysis ==========
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()

        # Group by cost_penalty and get mean values
        cost_penalty_groups = df.groupby('cost_penalty').agg({
            'test_final_wealth': 'mean',
            'test_trade_frequency': 'mean'
        }).reset_index()

        # Plot wealth on left y-axis
        color1 = 'tab:blue'
        ax2.scatter(cost_penalty_groups['cost_penalty'], cost_penalty_groups['test_final_wealth'],
                    color=color1, s=100, label='Final Wealth', zorder=3)
        ax2.set_xlabel('Cost Penalty', fontsize=11)
        ax2.set_ylabel('Test Final Wealth', color=color1, fontsize=11)
        ax2.tick_params(axis='y', labelcolor=color1)
        ax2.grid(True, alpha=0.3)

        # Plot trade frequency on right y-axis
        color2 = 'tab:orange'
        ax2_twin.scatter(cost_penalty_groups['cost_penalty'], cost_penalty_groups['test_trade_frequency'],
                         color=color2, s=100, marker='s', label='Trade Frequency', zorder=3)
        ax2_twin.set_ylabel('Trade Frequency', color=color2, fontsize=11)
        ax2_twin.tick_params(axis='y', labelcolor=color2)

        ax2.set_title('Effect of Cost Penalty', fontsize=12, fontweight='bold')

        # ========== PLOT 3: Alpha Analysis ==========
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()

        # Group by alpha and get mean values
        alpha_groups = df.groupby('alpha').agg({
            'test_final_wealth': 'mean',
            'test_trade_frequency': 'mean'
        }).reset_index()

        # Plot wealth on left y-axis
        color1 = 'tab:green'
        ax3.scatter(alpha_groups['alpha'], alpha_groups['test_final_wealth'],
                    color=color1, s=100, label='Final Wealth', zorder=3)
        ax3.set_xlabel('Alpha (No-Trade Threshold)', fontsize=11)
        ax3.set_ylabel('Test Final Wealth', color=color1, fontsize=11)
        ax3.tick_params(axis='y', labelcolor=color1)
        ax3.grid(True, alpha=0.3)

        # Plot trade frequency on right y-axis
        color2 = 'tab:red'
        ax3_twin.scatter(alpha_groups['alpha'], alpha_groups['test_trade_frequency'],
                         color=color2, s=100, marker='s', label='Trade Frequency', zorder=3)
        ax3_twin.set_ylabel('Trade Frequency', color=color2, fontsize=11)
        ax3_twin.tick_params(axis='y', labelcolor=color2)

        ax3.set_title('Effect of Alpha (No-Trade Threshold)', fontsize=12, fontweight='bold')

        # ========== PLOT 4: Improvement Threshold Analysis ==========
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()

        # Group by improvement_threshold and get mean values
        it_groups = df.groupby('improvement_threshold').agg({
            'test_final_wealth': 'mean',
            'test_trade_frequency': 'mean'
        }).reset_index()

        # Plot wealth on left y-axis
        color1 = 'tab:purple'
        ax4.scatter(it_groups['improvement_threshold'], it_groups['test_final_wealth'],
                    color=color1, s=100, label='Final Wealth', zorder=3)
        ax4.set_xlabel('Improvement Threshold', fontsize=11)
        ax4.set_ylabel('Test Final Wealth', color=color1, fontsize=11)
        ax4.tick_params(axis='y', labelcolor=color1)
        ax4.grid(True, alpha=0.3)

        # Plot trade frequency on right y-axis
        color2 = 'tab:brown'
        ax4_twin.scatter(it_groups['improvement_threshold'], it_groups['test_trade_frequency'],
                         color=color2, s=100, marker='s', label='Trade Frequency', zorder=3)
        ax4_twin.set_ylabel('Trade Frequency', color=color2, fontsize=11)
        ax4_twin.tick_params(axis='y', labelcolor=color2)

        ax4.set_title('Effect of Improvement Threshold', fontsize=12, fontweight='bold')

        # Adjust layout and save
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()
        plt.close()


def run_all_experiments(stocks, train_days=700, test_days=350, cost_models=None,
                        continue_from_training=False, parameter_grid=None,
                        save_results=True, n_jobs=-1):
    """
    Run all experiments with parallel execution.

    Args:
        n_jobs: Number of parallel jobs per cost model. -1 uses all available cores.
    """
    data_dict = prepare_stock_data(
        stocks=stocks,
        train_days=train_days,
        test_days=test_days
    )

    train_data = data_dict['train_price_relatives']
    test_data = data_dict['test_price_relatives']
    stock_names = data_dict['stock_names']

    bah_results = run_buy_and_hold(test_data)

    all_results = {}
    all_dfs = []

    for model_name, cost_model in cost_models.items():
        print(f"\n{'=' * 80}")
        print(f"Cost Model: {model_name}")
        print(f"{'=' * 80}\n")

        experiment = CostModelExperiment(
            cost_model=cost_model,
            model_name=model_name,
            stocks=stock_names,
            train_data=train_data,
            test_data=test_data
        )

        params = parameter_grid[model_name]

        experiment.grid_search(
            cost_penalties=params['cost_penalties'],
            alphas=params['alphas'],
            improvement_thresholds=params['improvement_thresholds'],
            continue_from_training=continue_from_training,
            verbose=True,
            n_jobs=n_jobs
        )

        all_results[model_name] = experiment
        all_dfs.append(experiment.get_results_dataframe())

        # Create plots
        if save_results:
            plot_dir = Path(r'C:\Users\rarustamyan\PycharmProjects\UniversalPortfolios\plots\portfolio\WithCosts\ONS')
            plot_dir.mkdir(parents=True, exist_ok=True)
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('$', '')
            experiment.plot_results(
                save_path=plot_dir / f'{safe_name}.png',
                continue_from_training=continue_from_training,
                bah_results=bah_results['daily_wealth']
            )

    # Combine all results
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save combined results
    if save_results:
        results_dir = Path(r'C:\Users\rarustamyan\PycharmProjects\UniversalPortfolios\result\portfolio')
        results_dir.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(results_dir / 'ons_costs_all_results.csv', index=False)
        print(f"\n\nResults saved to: {results_dir / 'ons_costs_all_results.csv'}")

    return all_results, combined_df


if __name__ == '__main__':
    # Configuration
    STOCKS = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]
    TRAIN_DAYS = 1500
    TEST_DAYS = 700
    SAVE_RESULTS = True
    N_JOBS = 16   # Use all available cores, or set to a specific number

    experiment_cost_models = {
        'Linear (0.1%)': LinearCost(gamma=0.001),
        'Linear (0.5%)': LinearCost(gamma=0.005),
        'Fixed Per Asset ($0.0005)': FixedCostPerAsset(cost_per_transaction=0.0005),
        'Fixed Per Rebalancing ($0.005)': FixedCostPerRebalancing(cost_per_transaction=0.005),
        'Quadratic (λ=0.01)': QuadraticCost(lambda_param=0.01),
    }

    linear_params = {
        'cost_penalties': [0, 0.1, 0.5, 1.0, 5.0, 50],
        'alphas': [0.0, 0.03, 0.05],
        'improvement_thresholds': [0.0, 0.0001, 0.0005, 0.001]
    }

    fixed_params = {
        'cost_penalties': [0, 50, 100, 200, 500, 1000],
        'alphas': [0.0, 0.02, 0.05, 0.10, 0.15],
        'improvement_thresholds': [0.0, 0.0001, 0.0005, 0.001, 0.002]
    }

    quadratic_params = {
        'cost_penalties': [0, 1, 5, 10, 20, 50, 100],
        'alphas': [0.0, 0.01, 0.03, 0.05, 0.08],
        'improvement_thresholds': [0.0, 0.0005, 0.001, 0.002, 0.005]
    }

    # Map cost models to parameter grids
    param_grids = {
        'Linear (0.1%)': linear_params,
        'Linear (0.5%)': linear_params,
        'Fixed Per Asset ($0.0005)': fixed_params,
        'Fixed Per Rebalancing ($0.005)': fixed_params,
        'Quadratic (λ=0.01)': quadratic_params,
    }

    experiments, results_df = run_all_experiments(
        stocks=STOCKS,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        cost_models=experiment_cost_models,
        parameter_grid=param_grids,
        save_results=SAVE_RESULTS,
        n_jobs=N_JOBS
    )