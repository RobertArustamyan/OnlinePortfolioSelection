import json
from pathlib import Path
from joblib import Parallel, delayed
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts
from algorithms.transactions.linear_cost import LinearCost
from algorithms.transactions.fixed_cost_per_asset import FixedCostPerAsset
from algorithms.transactions.fixed_cost_per_rebalancing import FixedCostPerRebalancing
from algorithms.transactions.quadratic_cost import QuadraticCost
from algorithms.transactions.interactive_brokers import InteractiveBrokersCostUSD
from utils.compare_strategies import compare_strategies
from utils.data_prep import prepare_stock_data
from benchmarks.portfolio.buy_and_hold import run_buy_and_hold
from utils.data_prep import NumpyEncoder
from utils.results_plotter import ResultsPlotter


class CostModelExperiment:
    def __init__(self, cost_model, model_name, stocks, train_data, test_data, train_prices=None, test_prices=None, initial_capital=None):
        self.cost_model = cost_model
        self.model_name = model_name
        self.stocks = stocks
        self.train_data = train_data
        self.test_data = test_data
        self.train_prices = train_prices
        self.test_prices = test_prices
        self.n_stocks = len(stocks)
        self.initial_capital = initial_capital

        # Compute ONS parameters
        T = len(train_data) + len(test_data)
        self.eta = (self.n_stocks ** 1.25) / (np.sqrt(T * np.log(self.n_stocks * T)))
        self.beta = 1.0 / (8 * (self.n_stocks ** 0.25) * np.sqrt(T * np.log(self.n_stocks * T)))
        self.delta = 0.8

        self.results = []
        self.detailed_results = []

    def run_single_experiment(self, cost_penalty, alpha, improvement_threshold,
                              continue_from_training=False):
        """Run a single experiment with given parameters"""
        # Algorithm initialization
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
        train_results = ons.simulate_trading(self.train_data, stock_prices_sequence=self.train_prices, verbose=False)

        # Testing phase
        if continue_from_training:
            test_daily_wealth = [train_results['daily_wealth'][-1]]
            test_wealth_current = train_results['daily_wealth'][-1]
        else:
            test_daily_wealth = [1.0]
            test_wealth_current = 1.0

        test_costs = []
        test_costs_dollars = []
        test_turnovers = []
        test_no_trades = 0
        test_portfolios = []
        test_traded_flags = []

        for day_idx, price_rel in enumerate(self.test_data):
            current_portfolio = ons.get_portfolio()
            prev_portfolio = ons.prev_portfolio.copy()

            # Store portfolio
            test_portfolios.append(current_portfolio.copy())

            # Update cost model with current wealth and today's prices
            if hasattr(self.cost_model, 'update_state'):
                self.cost_model.update_state(
                    wealth_fraction=test_wealth_current,
                    stock_prices=self.test_prices[day_idx]
                )

            # Compute cost and apply to wealth
            tc = self.cost_model.compute_cost(prev_portfolio, current_portfolio)

            dollar_cost = test_wealth_current * tc * self.initial_capital
            test_wealth_current *= (1 - tc)

            traded = dollar_cost > 0
            test_traded_flags.append(traded)
            if not traded:
                test_no_trades += 1

            test_costs.append(tc)
            test_costs_dollars.append(dollar_cost)

            # Compute turnover
            turnover = np.sum(np.abs(current_portfolio - prev_portfolio))
            test_turnovers.append(turnover)

            # Apply returns
            daily_return = np.dot(current_portfolio, price_rel)
            test_wealth_current *= daily_return
            test_daily_wealth.append(test_wealth_current)

            # Update algorithm
            ons.update(price_rel, current_portfolio)

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
            'test_cost_drag': sum(test_costs) * 100
        }

        detailed_result = {
            # Parameters
            'parameters': {
                'cost_penalty': cost_penalty,
                'alpha': alpha,
                'improvement_threshold': improvement_threshold,
                'model_name': self.model_name,
                'stocks': self.stocks,
                'n_stocks': self.n_stocks,
                'eta': self.eta,
                'beta': self.beta,
                'delta': self.delta,
            },
            # Training results
            'training': {
                'final_wealth': train_results['final_wealth'],
                'daily_wealth': train_results['daily_wealth'],
                'portfolios_used': train_results['portfolios_used'],
                'transaction_costs': train_results['transaction_costs'],
                'turnovers': train_results['turnovers'],
                'total_transaction_cost': train_results['total_transaction_cost'],
                'avg_turnover': train_results['avg_turnover'],
                'max_turnover': train_results['max_turnover'],
                'num_days': train_results['num_days'],
                'num_no_trades': train_results['num_no_trades'],
                'num_alpha_blocks': train_results['num_alpha_blocks'],
                'num_improvement_blocks': train_results['num_improvement_blocks'],
                'trade_frequency': train_results['trade_frequency'],
                'block_reasons': train_results['block_reasons']
            },
            # Testing results
            'testing': {
                'final_wealth': test_wealth_current,
                'daily_wealth': np.array(test_daily_wealth),
                'portfolios_used': np.array(test_portfolios),
                'transaction_costs': np.array(test_costs),
                'transaction_costs_dollars': np.array(test_costs_dollars),
                'turnovers': np.array(test_turnovers),
                'total_transaction_cost': sum(test_costs),
                'avg_turnover': np.mean(test_turnovers),
                'max_turnover': np.max(test_turnovers),
                'num_days': len(self.test_data),
                'num_no_trades': test_no_trades,
                'trade_frequency': 1 - (test_no_trades / len(self.test_data)),
                'traded_flags': test_traded_flags,
                'net_return': (test_wealth_current - 1.0) * 100,
                'cost_drag': sum(test_costs) * 100
            }
        }

        return result, detailed_result

    def grid_search(self, cost_penalties, alphas, improvement_thresholds,
                    continue_from_training=False, verbose=True, n_jobs=-1):
        """
        Run grid search with parallel execution.

        n_jobs: Number of parallel jobs. -1 uses all available cores.
        # TODO: add description of other parameters
        """
        if verbose:
            print(f"Grid Search for {self.model_name}")
            print(f"Testing {len(cost_penalties)} × {len(alphas)} × {len(improvement_thresholds)} "
                  f"= {len(cost_penalties) * len(alphas) * len(improvement_thresholds)} combinations")
            print(f"n_jobs={n_jobs if n_jobs != -1 else 'all cores'}\n")

        # Create all parameter combinations
        param_combinations = [
            (cp, a, it)
            for cp in cost_penalties
            for a in alphas
            for it in improvement_thresholds
        ]

        # Run experiments in parallel
        all_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(self.run_single_experiment)(
                cost_penalty=cp,
                alpha=a,
                improvement_threshold=it,
                continue_from_training=continue_from_training
            )
            for cp, a, it in param_combinations
        )

        self.results = [r[0] for r in all_results]
        self.detailed_results = [r[1] for r in all_results]

        if verbose:
            print(f"\nCompleted {len(self.results)} experiments")
            best_result = max(self.results, key=lambda x: x['test_final_wealth'])
            print(f"\nBest result:")
            print(f" Cost Penalty: {best_result['cost_penalty']}")
            print(f" Alpha: {best_result['alpha']}")
            print(f" Improvement Threshold: {best_result['improvement_threshold']}")
            print(f" Test Final Wealth: {best_result['test_final_wealth']:.4f}")
            print(f" Trade Frequency: {best_result['test_trade_frequency']:.2%}\n")

        return self.results

    def get_results_dataframe(self):
        """Convert results to pandas DataFrame"""
        df = pd.DataFrame(self.results)
        df['model'] = self.model_name
        return df

    def get_detailed_results(self):
        """Return results for json export"""
        return self.detailed_results

    def create_plotter(self):
        return ResultsPlotter(self.results, self.model_name)


def run_all_experiments(stocks, train_days=700, test_days=350, cost_models=None,
                        continue_from_training=False, parameter_grid=None,
                        save_results=True, n_jobs=-1, initial_capital=None):
    """
    Run all experiments with parallel execution.

    Args:
        n_jobs: Number of parallel jobs per cost model. -1 uses all available cores.
        initial_capital: Initial capital for metadata
    """
    data_dict = prepare_stock_data(
        stocks=stocks,
        train_days=train_days,
        test_days=test_days
    )

    train_data = data_dict['train_price_relatives']
    test_data = data_dict['test_price_relatives']
    train_prices = data_dict['train_actual_prices']
    test_prices = data_dict['test_actual_prices']

    stock_names = data_dict['stock_names']

    bah_results = run_buy_and_hold(test_data)

    # Create timestamped results directory
    if save_results:
        timestamp = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        folder_name = f"train{train_days}_test{test_days}_stocks{len(stocks)}_{timestamp}"

        base_dir = Path(r'C:\Users\rarustamyan\PycharmProjects\UniversalPortfolios\results\portfolio\WithCosts\ONS')
        results_dir = base_dir / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"Results will be saved to: {results_dir}")
        print(f"{'=' * 80}\n")

        # Save metadata
        metadata = {
            'run_info': {
                'timestamp': timestamp,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'experiment_settings': {
                'stocks': stocks,
                'n_stocks': len(stocks),
                'train_days': train_days,
                'test_days': test_days,
                'n_jobs': n_jobs,
                'initial_capital': initial_capital,
                'continue_from_training': continue_from_training
            },
            'parameter_grids': parameter_grid,
            'cost_models': list(cost_models.keys())
        }

        metadata_path = results_dir / 'experiment_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}\n")

    all_results = {}
    all_dfs = []
    all_detailed_results = {}

    for model_name, cost_model in cost_models.items():
        print(f"\n{'=' * 80}")
        print(f"Cost Model: {model_name}")
        print(f"{'=' * 80}\n")

        experiment = CostModelExperiment(
            cost_model=cost_model,
            model_name=model_name,
            stocks=stock_names,
            train_data=train_data,
            test_data=test_data,
            train_prices=train_prices,
            test_prices=test_prices,
            initial_capital=initial_capital
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
        all_detailed_results[model_name] = experiment.get_detailed_results()

        # Create plots using the new ResultsPlotter class
        if save_results:
            plotter = experiment.create_plotter()
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('$', '')

            # Main 2x2 plot
            plotter.plot_all(
                save_path=results_dir / f'{safe_name}.png',
                bah_results=bah_results['daily_wealth']
            )

            # Get top 5 results for comparison plot
            df_sorted = plotter.get_dataframe().sort_values('test_final_wealth', ascending=False)
            top_5 = df_sorted.head(5)
            best_result = top_5.iloc[0]

            # Find the best configuration's detailed result
            best_detail = None
            for detail in all_detailed_results[model_name]:
                if (detail['parameters']['cost_penalty'] == best_result['cost_penalty'] and
                        detail['parameters']['alpha'] == best_result['alpha'] and
                        detail['parameters']['improvement_threshold'] == best_result['improvement_threshold']):
                    best_detail = detail
                    break

            # Get initial portfolio from the BEST configuration
            if best_detail:
                initial_portfolio = best_detail['testing']['portfolios_used'][0]

                # Run B&H with the best configuration's initial portfolio
                modified_bah = run_buy_and_hold(test_data, initial_weights=initial_portfolio)
                modified_bah_list = [modified_bah['daily_wealth']]

                # Extract trade dates (where block_reason is 'traded')
                trade_dates = [i for i, traded in enumerate(best_detail['testing']['traded_flags']) if
                               traded]
                trade_costs_dollars = [best_detail['testing']['transaction_costs_dollars'][i]
                                       for i, traded in enumerate(best_detail['testing']['traded_flags']) if traded]
            else:
                modified_bah_list = None
                trade_dates = None
                trade_costs_dollars = None

            # Create comparison plot
            plotter.plot_wealth_comparison(
                save_path=results_dir / f'{safe_name}_comparison.png',
                # bah_results=bah_results['daily_wealth'],
                modified_bah_results=modified_bah_list,
                trade_dates=trade_dates,
                trade_costs_dollars=trade_costs_dollars
            )

    # Combine all results
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save results
    if save_results:
        # Save CSV (summary)
        combined_df.to_csv(results_dir / 'ons_costs_all_results.csv', index=False)
        print(f"\n\nCSV results saved to: {results_dir / 'ons_costs_all_results.csv'}")

        # Save detailed JSON
        detailed_json = {
            'metadata': metadata,
            'buy_and_hold': {
                'final_wealth': float(bah_results['final_wealth']),
                'daily_wealth': bah_results['daily_wealth'].tolist()
            },
            'experiments': all_detailed_results
        }

        json_path = results_dir / 'ons_costs_detailed_results.json'
        with open(json_path, 'w') as f:
            json.dump(detailed_json, f, cls=NumpyEncoder, indent=2)
        print(f"JSON detailed results saved to: {json_path}")

        print(f"\n{'=' * 80}")
        print(f"All results saved in: {results_dir}")
        print(f"{'=' * 80}\n")

    return all_results, combined_df


if __name__ == '__main__':
    # Configuration
    STOCKS = ["GOOGL", "AMC", "NVDA", "AAPL",]
    TRAIN_DAYS = 200
    TEST_DAYS = 500
    SAVE_RESULTS = True
    N_JOBS = 16

    initial_capital = 1000

    price_to_k = initial_capital / 1000
    experiment_cost_models = {
        # f'IB Tiered (${price_to_k}k)': InteractiveBrokersCostUSD(
        #     initial_capital=initial_capital,
        #     pricing_type="tiered"
        # ),
        f'IB Fixed (${price_to_k}k)': InteractiveBrokersCostUSD(
            initial_capital=initial_capital,
            pricing_type="fixed"
        ),
    }

    ib_params = {
        'cost_penalties': [0, 0.5, 10, 50, 100, 200, 500],
        'alphas': [0.0, 0.01, 0.02, 0.03, 0.05],
        # 'improvement_thresholds': [0.0, 0.0001, 0.0005, 0.001, 0.002]
        'improvement_thresholds': [0.0]
    }

    param_grids = {
        f'IB Tiered (${price_to_k}k)': ib_params,
        f'IB Fixed (${price_to_k}k)': ib_params,
    }

    experiments, results_df = run_all_experiments(
        stocks=STOCKS,
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        cost_models=experiment_cost_models,
        parameter_grid=param_grids,
        save_results=SAVE_RESULTS,
        n_jobs=N_JOBS,
        initial_capital=initial_capital
    )