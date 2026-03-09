import copy
import jsonimport copy
from pathlib import Path
from joblib import Parallel, delayed
from datetime import datetime
import json

import numpy as np
import pandas as pd

from algorithms.portfolio.barrons_costs import BarronsCosts
from benchmarks.portfolio.buy_and_hold import run_buy_and_hold
from algorithms.transactions.interactive_brokers import InteractiveBrokersCostUSD
from utils.data_prep import prepare_stock_data_3split


class BarronsCostModelExperiment3Split:
    def __init__(self, cost_model, model_name, stocks, train_data, val_data, test_data,
                 train_prices=None, val_prices=None, test_prices=None, initial_capital=None):
        self.cost_model = cost_model
        self.model_name = model_name
        self.stocks = stocks
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_prices = train_prices
        self.val_prices = val_prices
        self.test_prices = test_prices
        self.n_stocks = len(stocks)
        self.initial_capital = initial_capital

        # Compute total sequence length (used by BarronsCosts internally)
        T = len(train_data) + len(val_data) + len(test_data)
        self.T = T

        self.all_results = []  # All parameter combinations with train+val results
        self.best_params = None
        self.retrain_results = None
        self.test_results = None

    def run_single_training(self, eta, beta, cost_penalty, alpha, optimization_method):
        """
        Train and validate one configuration.
        Returns both training and validation results.
        """
        cost_model_train = copy.deepcopy(self.cost_model)

        # ========== TRAINING PHASE ==========
        barrons_train = BarronsCosts(
            n_stocks=self.n_stocks,
            T=self.T,
            beta=beta,
            eta=eta,
            cost_model=cost_model_train,
            cost_penalty=cost_penalty,
            alpha=alpha,
            optimization_method=optimization_method
        )

        train_results = barrons_train.simulate_trading(
            self.train_data,
            stock_prices_sequence=self.train_prices,
            verbose=False
        )

        # ========== VALIDATION PHASE ==========
        cost_model_val = copy.deepcopy(self.cost_model)
        barrons_val = BarronsCosts(
            n_stocks=self.n_stocks,
            T=self.T,
            beta=beta,
            eta=eta,
            cost_model=cost_model_val,
            cost_penalty=cost_penalty,
            alpha=alpha,
            optimization_method=optimization_method
        )

        # Initialize with learned state from training
        barrons_val.A = barrons_train.A.copy()
        barrons_val.eta_t = barrons_train.eta_t.copy()
        barrons_val.x_t = barrons_train.x_t.copy()
        barrons_val.prev_portfolio = barrons_train.prev_portfolio.copy()

        # Run validation with fresh capital
        val_wealth = 1.0
        val_daily_wealth = [val_wealth]
        val_costs = []
        val_turnovers = []
        val_no_trades = 0

        val_portfolios = []
        for day_idx, price_rel in enumerate(self.val_data):
            # Update cost model state BEFORE getting portfolio
            if hasattr(cost_model_val, 'update_state'):
                cost_model_val.update_state(
                    wealth_fraction=val_wealth,
                    stock_prices=self.val_prices[day_idx]
                )

            # Now get portfolio (may need cost model for objective computation)
            current_portfolio = barrons_val.get_portfolio()
            val_portfolios.append(current_portfolio)
            prev_portfolio = barrons_val.prev_portfolio.copy()

            # Add to history before update
            barrons_val.portfolios_used.append(current_portfolio.copy())

            # Compute cost
            tc = cost_model_val.compute_cost(prev_portfolio, current_portfolio)
            val_wealth *= (1 - tc)

            if tc == 0:
                val_no_trades += 1

            val_costs.append(tc)
            turnover = np.sum(np.abs(current_portfolio - prev_portfolio))
            val_turnovers.append(turnover)

            # Apply returns
            daily_return = np.dot(current_portfolio, price_rel)
            val_wealth *= daily_return
            val_daily_wealth.append(val_wealth)

            # Update algorithm
            barrons_val.update(price_rel)
            barrons_val.prev_portfolio = current_portfolio.copy()

        # Return results for this parameter combination
        result = {
            'parameters': {
                'eta': eta,
                'beta': beta,
                'cost_penalty': cost_penalty,
                'alpha': alpha,
                'optimization_method': optimization_method,
            },
            'training': {
                'final_wealth': train_results['final_wealth'],
                'daily_wealth': train_results['daily_wealth'].tolist(),
                'portfolios_used': train_results['portfolios_used'].tolist(),
                'transaction_costs': train_results['transaction_costs'].tolist(),
                'total_transaction_cost': train_results['total_transaction_cost'],
                'avg_turnover': train_results['avg_turnover'],
                'trade_frequency': train_results['trade_frequency'],
                'num_days': train_results['num_days']
            },
            'validation': {
                'final_wealth': val_wealth,
                'daily_wealth': val_daily_wealth,
                'portfolios_used': [p.tolist() for p in val_portfolios],
                'transaction_costs': val_costs,
                'total_transaction_cost': sum(val_costs),
                'avg_turnover': np.mean(val_turnovers),
                'trade_frequency': 1 - (val_no_trades / len(self.val_data)),
                'num_days': len(self.val_data)
            }
        }

        return result

    def grid_search(self, etas, betas, cost_penalties, alphas, optimization_methods,
                    verbose=True, n_jobs=-1):
        """
        Run grid search: train all combinations, validate, select best.
        """
        if verbose:
            print(f"Grid Search for {self.model_name}")
            total_combos = (len(etas) * len(betas) * len(cost_penalties)
                            * len(alphas) * len(optimization_methods))
            print(
                f"Testing {len(etas)} × {len(betas)} × {len(cost_penalties)} × "
                f"{len(alphas)} × {len(optimization_methods)} = {total_combos} combinations"
            )
            print(f"n_jobs={n_jobs if n_jobs != -1 else 'all cores'}\n")

        # Create all parameter combinations (eta, beta first)
        param_combinations = [
            (eta, beta, cp, a, om)
            for eta in etas
            for beta in betas
            for cp in cost_penalties
            for a in alphas
            for om in optimization_methods
        ]

        # Run experiments in parallel
        all_results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(self.run_single_training)(eta, beta, cp, a, om)
            for eta, beta, cp, a, om in param_combinations
        )

        self.all_results = all_results

        # Find best based on validation performance
        best_result = max(all_results, key=lambda x: x['validation']['final_wealth'])
        self.best_params = best_result['parameters']

        if verbose:
            print(f"\nCompleted {len(all_results)} experiments")
            print(f"\nBest result (selected on VALIDATION):")
            print(f"  Eta: {self.best_params['eta']}")
            print(f"  Beta: {self.best_params['beta']}")
            print(f"  Cost Penalty: {self.best_params['cost_penalty']}")
            print(f"  Alpha: {self.best_params['alpha']}")
            print(f"  Optimization Method: {self.best_params['optimization_method']}")
            print(f"  Train Final Wealth: {best_result['training']['final_wealth']:.4f}")
            print(f"  Validation Final Wealth: {best_result['validation']['final_wealth']:.4f}")

        return self.best_params

    def run_test(self, best_params, verbose=True):
        """
        Retrain on train+val with best hyperparams, then test.
        """
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"RETRAINING ON TRAIN+VAL, THEN TESTING")
            print(f"{'=' * 80}")

        # Combine train and val data
        combined_data = np.vstack([self.train_data, self.val_data])
        combined_prices = (
            np.vstack([self.train_prices, self.val_prices])
            if self.train_prices is not None else None
        )

        if verbose:
            print(f"\nRetraining with best hyperparameters on {len(combined_data)} days (train+val)...")
            for key, value in best_params.items():
                print(f"  {key}: {value}")
            print()

        # ========== RETRAIN ON TRAIN+VAL ==========
        cost_model_retrain = copy.deepcopy(self.cost_model)
        barrons_retrain = BarronsCosts(
            n_stocks=self.n_stocks,
            T=self.T,
            beta=best_params['beta'],
            eta=best_params['eta'],
            cost_model=cost_model_retrain,
            cost_penalty=best_params['cost_penalty'],
            alpha=best_params['alpha'],
            optimization_method=best_params['optimization_method']
        )

        retrain_results = barrons_retrain.simulate_trading(
            combined_data,
            stock_prices_sequence=combined_prices,
            verbose=False
        )

        self.retrain_results = {
            'final_wealth': retrain_results['final_wealth'],
            'daily_wealth': retrain_results['daily_wealth'],
            'portfolios_used': retrain_results['portfolios_used'],
            'transaction_costs': retrain_results['transaction_costs'],
            'turnovers': retrain_results['turnovers'],
            'total_transaction_cost': retrain_results['total_transaction_cost'],
            'avg_turnover': retrain_results['avg_turnover'],
            'trade_frequency': retrain_results['trade_frequency'],
            'num_days': retrain_results['num_days']
        }

        if verbose:
            print(f"Retraining complete:")
            print(f"  Final Wealth: {retrain_results['final_wealth']:.4f}")
            print(f"  Trade Frequency: {retrain_results['trade_frequency']:.2%}\n")

        # ========== TEST PHASE ==========
        if verbose:
            print(f"Testing on {len(self.test_data)} days...\n")

        cost_model_test = copy.deepcopy(self.cost_model)
        barrons_test = BarronsCosts(
            n_stocks=self.n_stocks,
            T=self.T,
            beta=best_params['beta'],
            eta=best_params['eta'],
            cost_model=cost_model_test,
            cost_penalty=best_params['cost_penalty'],
            alpha=best_params['alpha'],
            optimization_method=best_params['optimization_method']
        )

        # Load learned state from retrained model
        barrons_test.A = barrons_retrain.A.copy()
        barrons_test.eta_t = barrons_retrain.eta_t.copy()
        barrons_test.x_t = barrons_retrain.x_t.copy()
        barrons_test.prev_portfolio = barrons_retrain.prev_portfolio.copy()

        # Run test with fresh capital
        test_wealth = 1.0
        test_daily_wealth = [test_wealth]
        test_costs = []
        test_costs_dollars = []
        test_turnovers = []
        test_no_trades = 0
        test_portfolios = []
        test_traded_flags = []

        for day_idx, price_rel in enumerate(self.test_data):
            # Update cost model state BEFORE getting portfolio
            if hasattr(cost_model_test, 'update_state'):
                cost_model_test.update_state(
                    wealth_fraction=test_wealth,
                    stock_prices=self.test_prices[day_idx]
                )

            # Now get portfolio
            current_portfolio = barrons_test.get_portfolio()
            prev_portfolio = barrons_test.prev_portfolio.copy()

            test_portfolios.append(current_portfolio.copy())

            # Add to history before update
            barrons_test.portfolios_used.append(current_portfolio.copy())

            # Compute cost
            tc = cost_model_test.compute_cost(prev_portfolio, current_portfolio)
            dollar_cost = test_wealth * tc * self.initial_capital
            test_wealth *= (1 - tc)

            traded = dollar_cost > 0
            test_traded_flags.append(traded)
            if not traded:
                test_no_trades += 1

            test_costs.append(tc)
            test_costs_dollars.append(dollar_cost)

            turnover = np.sum(np.abs(current_portfolio - prev_portfolio))
            test_turnovers.append(turnover)

            # Apply returns
            daily_return = np.dot(current_portfolio, price_rel)
            test_wealth *= daily_return
            test_daily_wealth.append(test_wealth)

            # Update algorithm
            barrons_test.update(price_rel)
            barrons_test.prev_portfolio = current_portfolio.copy()

        self.test_results = {
            'final_wealth': test_wealth,
            'daily_wealth': test_daily_wealth,
            'portfolios_used': test_portfolios,
            'transaction_costs': test_costs,
            'transaction_costs_dollars': test_costs_dollars,
            'turnovers': test_turnovers,
            'total_transaction_cost': sum(test_costs),
            'avg_turnover': np.mean(test_turnovers),
            'trade_frequency': 1 - (test_no_trades / len(self.test_data)),
            'traded_flags': test_traded_flags,
            'num_days': len(self.test_data),
            'net_return_pct': (test_wealth - 1.0) * 100,
            'cost_drag_pct': sum(test_costs) * 100,
            'barrons_initial_portfolio': barrons_test.prev_portfolio.tolist()
        }

        if verbose:
            print(f"TEST RESULTS:")
            print(f"  Final Wealth: {test_wealth:.4f} ({self.test_results['net_return_pct']:.2f}%)")
            print(f"  Total Cost: {sum(test_costs):.6f} ({self.test_results['cost_drag_pct']:.4f}% drag)")
            print(f"  Trade Frequency: {self.test_results['trade_frequency']:.2%}")
            print(f"  Avg Turnover: {np.mean(test_turnovers):.4f}")
            print(f"{'=' * 80}\n")

        return self.test_results


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        return super().default(obj)


def save_experiment_results(results_dir, experiments, metadata):
    """
    Save experiment results in a clean structure.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save metadata
    metadata_path = results_dir / 'experiment_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # 2. Save all_results.csv (hyperparameter combinations summary)
    all_results_list = []
    for model_name, experiment in experiments.items():
        for result in experiment.all_results:
            row = {
                'model': model_name,
                **result['parameters'],   # eta, beta, cost_penalty, alpha, optimization_method
                'train_final_wealth': result['training']['final_wealth'],
                'train_total_cost': result['training']['total_transaction_cost'],
                'train_trade_freq': result['training']['trade_frequency'],
                'val_final_wealth': result['validation']['final_wealth'],
                'val_total_cost': result['validation']['total_transaction_cost'],
                'val_trade_freq': result['validation']['trade_frequency']
            }
            all_results_list.append(row)

    df_all = pd.DataFrame(all_results_list)
    all_results_path = results_dir / 'all_results.csv'
    df_all.to_csv(all_results_path, index=False)
    print(f"All hyperparameter results saved to: {all_results_path}")

    # 3. Save detailed_results.json (complete data)
    detailed_results = {}
    for model_name, experiment in experiments.items():
        detailed_results[model_name] = {
            'best_parameters': experiment.best_params,
            'all_parameter_combinations': experiment.all_results,
            'retrain_on_train_val': {
                'final_wealth': experiment.retrain_results['final_wealth'],
                'daily_wealth': experiment.retrain_results['daily_wealth'].tolist(),
                'portfolios_used': experiment.retrain_results['portfolios_used'].tolist(),
                'transaction_costs': experiment.retrain_results['transaction_costs'].tolist(),
                'turnovers': experiment.retrain_results['turnovers'].tolist(),
                'total_transaction_cost': experiment.retrain_results['total_transaction_cost'],
                'avg_turnover': experiment.retrain_results['avg_turnover'],
                'trade_frequency': experiment.retrain_results['trade_frequency'],
                'num_days': experiment.retrain_results['num_days']
            },
            'test': {
                'final_wealth': experiment.test_results['final_wealth'],
                'daily_wealth': experiment.test_results['daily_wealth'],
                'portfolios_used': experiment.test_results['portfolios_used'],
                'transaction_costs': experiment.test_results['transaction_costs'],
                'transaction_costs_dollars': experiment.test_results['transaction_costs_dollars'],
                'turnovers': experiment.test_results['turnovers'],
                'total_transaction_cost': experiment.test_results['total_transaction_cost'],
                'avg_turnover': experiment.test_results['avg_turnover'],
                'trade_frequency': experiment.test_results['trade_frequency'],
                'traded_flags': experiment.test_results['traded_flags'],
                'num_days': experiment.test_results['num_days'],
                'net_return_pct': experiment.test_results['net_return_pct'],
                'cost_drag_pct': experiment.test_results['cost_drag_pct'],
                'barrons_initial_portfolio': experiment.test_results['barrons_initial_portfolio']
            }
        }

    detailed_path = results_dir / 'detailed_results.json'
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, cls=NumpyEncoder, indent=2)
    print(f"Detailed results saved to: {detailed_path}")

    print(f"\n{'=' * 80}")
    print(f"All results saved in: {results_dir}")
    print(f"{'=' * 80}\n")

    return results_dir


def run_3split_experiment(stocks, train_start_date, train_end_date, val_end_date, test_end_date,
                          cost_models, parameter_grid, results_base_dir, save_results=True,
                          n_jobs=-1, initial_capital=1000):
    """
    Run complete 3-way split experiment for BARRONS: Train → Validate → Test.
    """
    # Prepare data
    print(f"Preparing data...")
    data_dict = prepare_stock_data_3split(
        stocks=stocks,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        test_end_date=test_end_date
    )

    train_data = data_dict['train_price_relatives']
    val_data = data_dict['val_price_relatives']
    test_data = data_dict['test_price_relatives']

    train_prices = data_dict['train_actual_prices']
    val_prices = data_dict['val_actual_prices']
    test_prices = data_dict['test_actual_prices']

    stock_names = data_dict['stock_names']

    # Compute benchmarks
    print(f"Computing benchmarks...")

    # Uniform Buy-and-Hold
    bah_uniform_results = run_buy_and_hold(test_data)

    # Index benchmarks
    nasdaq_test_results = None
    sp500_test_results = None
    if 'benchmark_test_relatives' in data_dict:
        if data_dict['benchmark_test_relatives'].get('NASDAQ') is not None:
            nasdaq_test_results = run_buy_and_hold(data_dict['benchmark_test_relatives']['NASDAQ'])
        if data_dict['benchmark_test_relatives'].get('SP500') is not None:
            sp500_test_results = run_buy_and_hold(data_dict['benchmark_test_relatives']['SP500'])

    # Create results directory
    if save_results:
        timestamp = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        folder_name = f"barrons_3split_stocks{len(stocks)}_{timestamp}"
        results_dir = Path(results_base_dir) / folder_name
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"Results will be saved to: {results_dir}")
        print(f"{'=' * 80}\n")

    # Run experiments for each cost model
    all_experiments = {}

    for model_name, cost_model in cost_models.items():
        print(f"\n{'=' * 80}")
        print(f"Cost Model: {model_name}")
        print(f"{'=' * 80}\n")

        experiment = BarronsCostModelExperiment3Split(
            cost_model=cost_model,
            model_name=model_name,
            stocks=stock_names,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_prices=train_prices,
            val_prices=val_prices,
            test_prices=test_prices,
            initial_capital=initial_capital
        )

        params = parameter_grid[model_name]

        # Grid search (train + validate, select best)
        best_params = experiment.grid_search(
            etas=params['etas'],
            betas=params['betas'],
            cost_penalties=params['cost_penalties'],
            alphas=params['alphas'],
            optimization_methods=params['optimization_methods'],
            verbose=True,
            n_jobs=n_jobs
        )

        # Retrain on train+val, then test
        test_results = experiment.run_test(best_params, verbose=True)

        all_experiments[model_name] = experiment

    # After running experiments, compute BARRONS initial portfolio BAH
    print(f"Computing BARRONS initial portfolio Buy-and-Hold benchmark...")
    bah_barrons_initial_results = {}
    for model_name, experiment in all_experiments.items():
        barrons_initial_portfolio = np.array(experiment.test_results['barrons_initial_portfolio'])
        bah_barrons_result = run_buy_and_hold(test_data, initial_weights=barrons_initial_portfolio)
        bah_barrons_initial_results[model_name] = bah_barrons_result

        print(f"  {model_name}:")
        print(f"    BARRONS Initial Portfolio: {barrons_initial_portfolio}")
        print(f"    BAH (BARRONS Initial) Final Wealth: {bah_barrons_result['final_wealth']:.4f}")

    # Save all results
    if save_results:
        metadata = {
            'run_info': {
                'timestamp': timestamp,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'experiment_settings': {
                'stocks': stocks,
                'n_stocks': len(stocks),
                'train_start': train_start_date,
                'train_end': train_end_date,
                'val_end': val_end_date,
                'test_end': test_end_date,
                'train_days': len(train_data),
                'val_days': len(val_data),
                'test_days': len(test_data),
                'initial_capital': initial_capital,
                'n_jobs': n_jobs
            },
            'parameter_grids': parameter_grid,
            'cost_models': list(cost_models.keys()),
            'benchmarks': {
                'buy_and_hold_uniform': {
                    'final_wealth': float(bah_uniform_results['final_wealth']),
                    'daily_wealth': bah_uniform_results['daily_wealth'].tolist()
                },
                'buy_and_hold_barrons_initial': {
                    model_name: {
                        'initial_portfolio': bah_barrons_initial_results[model_name]['initial_weights'].tolist(),
                        'final_wealth': float(bah_barrons_initial_results[model_name]['final_wealth']),
                        'daily_wealth': bah_barrons_initial_results[model_name]['daily_wealth'].tolist()
                    }
                    for model_name in all_experiments.keys()
                },
                'nasdaq': {
                    'available': nasdaq_test_results is not None,
                    'final_wealth': float(nasdaq_test_results['final_wealth']) if nasdaq_test_results else None,
                    'daily_wealth': nasdaq_test_results['daily_wealth'].tolist() if nasdaq_test_results else None
                },
                'sp500': {
                    'available': sp500_test_results is not None,
                    'final_wealth': float(sp500_test_results['final_wealth']) if sp500_test_results else None,
                    'daily_wealth': sp500_test_results['daily_wealth'].tolist() if sp500_test_results else None
                }
            }
        }

        save_experiment_results(results_dir, all_experiments, metadata)

    # Print final summary
    print(f"\n{'=' * 80}")
    print(f"FINAL TEST RESULTS SUMMARY")
    print(f"{'=' * 80}")

    print(f"\nBENCHMARKS:")
    print(f"  Uniform BAH: {bah_uniform_results['final_wealth']:.4f}")
    for model_name in all_experiments.keys():
        print(f"  BAH (BARRONS Initial Portfolio): {bah_barrons_initial_results[model_name]['final_wealth']:.4f}")
    if nasdaq_test_results:
        print(f"  NASDAQ: {nasdaq_test_results['final_wealth']:.4f}")
    if sp500_test_results:
        print(f"  S&P 500: {sp500_test_results['final_wealth']:.4f}")

    print(f"\nBARRONS RESULTS:")
    for model_name, experiment in all_experiments.items():
        print(f"\n{model_name}:")
        print(f"  Best Parameters:")
        for key, value in experiment.best_params.items():
            print(f"    {key}: {value}")
        print(f"  Test Performance:")
        print(f"    Final Wealth: {experiment.test_results['final_wealth']:.4f}")
        print(f"    Return: {experiment.test_results['net_return_pct']:.2f}%")
        print(f"    Cost Drag: {experiment.test_results['cost_drag_pct']:.4f}%")
        print(f"    Trade Frequency: {experiment.test_results['trade_frequency']:.2%}")

    return all_experiments


if __name__ == "__main__":

    # Date ranges
    TRAIN_START_DATE = "2021-02-01"
    TRAIN_END_DATE = "2023-02-01"
    VAL_END_DATE = "2024-02-01"
    TEST_END_DATE = "2026-02-01"

    # Execution settings
    SAVE_RESULTS = True
    N_JOBS = 19
    INITIAL_CAPITAL = 10000

    # Results directory
    RESULTS_BASE_DIR = '/home/robert/PycharmProjects/OnlinePortfolioSelection/results/portfolio/WithCosts/BARRONS'

    # ========== COST MODELS ==========
    price_to_k = INITIAL_CAPITAL / 1000
    cost_models = {
        f'IB Fixed (${price_to_k}k)': InteractiveBrokersCostUSD(
            initial_capital=INITIAL_CAPITAL,
            pricing_type="fixed"
        ),
    }

    # ========== HYPERPARAMETER GRIDS ==========
    # BARRONS doesn't have delta parameter, but has optimization_method
    barrons_params = {

        'etas': [0.1, 0.5, 1],
        'betas': [0.1, 0.25, 0.5],
        'cost_penalties': [0, 10, 1000],
        'alphas': [0.0, 0.01, 0.05],
        'optimization_methods': ['sqp']  # Two optimization approaches
    }

    parameter_grids = {
        f'IB Fixed (${price_to_k}k)': barrons_params,
    }

    # ========== EXPERIMENT CONFIGURATIONS ==========
    experiment_configs = [
        {
            'name': 'Tech Giants (5 stocks)',
            'stocks': ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        },
        {
            'name': 'High Growth Tech (5 stocks)',
            'stocks': ["NVDA", "TSLA", "AMD", "PLTR", "SNOW"]
        }
    ]

    # ========== RUN ALL EXPERIMENTS ==========
    all_experiments_results = {}

    for i, config in enumerate(experiment_configs, 1):
        print("\n" + "=" * 100)
        print(f"EXPERIMENT {i}/{len(experiment_configs)}: {config['name']}")
        print("=" * 100 + "\n")

        STOCKS = config['stocks']

        # Create cost model
        price_to_k = INITIAL_CAPITAL / 1000
        cost_models = {
            f"IB Fixed (${price_to_k}k)": InteractiveBrokersCostUSD(
                initial_capital=INITIAL_CAPITAL,
                pricing_type="fixed"
            ),
        }

        parameter_grids = {
            f"IB Fixed (${price_to_k}k)": barrons_params,
        }

        # Run experiment
        experiments = run_3split_experiment(
            stocks=STOCKS,
            train_start_date=TRAIN_START_DATE,
            train_end_date=TRAIN_END_DATE,
            val_end_date=VAL_END_DATE,
            test_end_date=TEST_END_DATE,
            cost_models=cost_models,
            parameter_grid=parameter_grids,
            results_base_dir=RESULTS_BASE_DIR,
            save_results=SAVE_RESULTS,
            n_jobs=N_JOBS,
            initial_capital=INITIAL_CAPITAL
        )

        all_experiments_results[config['name']] = experiments

        print(f"\n✓ Experiment {i} complete: {config['name']}\n")
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend  # optuna >= 3.6

from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts

optuna.logging.set_verbosity(optuna.logging.WARNING)


class CostModelExperiment3SplitOptuna:
    def __init__(self, cost_model, model_name, stocks, train_data, val_data, test_data,
                 train_prices=None, val_prices=None, test_prices=None, initial_capital=None):
        self.cost_model = cost_model
        self.model_name = model_name
        self.stocks = stocks
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_prices = train_prices
        self.val_prices = val_prices
        self.test_prices = test_prices
        self.n_stocks = len(stocks)
        self.initial_capital = initial_capital

        # Compute ONS parameters based on TOTAL sequence length
        T = len(train_data) + len(val_data) + len(test_data)
        self.eta = (self.n_stocks ** 1.25) / (np.sqrt(T * np.log(self.n_stocks * T)))
        self.beta = 1.0 / (8 * (self.n_stocks ** 0.25) * np.sqrt(T * np.log(self.n_stocks * T)))

        self.study = None
        self.best_params = None
        self.all_results = []
        self.retrain_results = None
        self.test_results = None

    # ------------------------------------------------------------------
    # Core single-trial evaluation
    # ------------------------------------------------------------------
    def run_single_training(self, delta, cost_penalty, alpha):
        """Train on train split, validate on val split. Returns result dict."""

        cost_model_train = copy.deepcopy(self.cost_model)

        # ── TRAINING ──────────────────────────────────────────────────
        ons_train = OnlineNewtonStepCosts(
            n_stocks=self.n_stocks,
            eta=self.eta,
            beta=self.beta,
            delta=delta,
            cost_model=cost_model_train,
            cost_penalty=cost_penalty,
            alpha=alpha,
        )
        train_results = ons_train.simulate_trading(
            self.train_data,
            stock_prices_sequence=self.train_prices,
            verbose=False,
        )

        # ── VALIDATION ────────────────────────────────────────────────
        cost_model_val = copy.deepcopy(self.cost_model)
        ons_val = OnlineNewtonStepCosts(
            n_stocks=self.n_stocks,
            eta=self.eta,
            beta=self.beta,
            delta=delta,
            cost_model=cost_model_val,
            cost_penalty=cost_penalty,
            alpha=alpha,
        )

        # Transfer learned state from training
        ons_val.A = ons_train.A.copy()
        ons_val.b = ons_train.b.copy()
        ons_val.p_t = ons_train.p_t.copy()
        ons_val.prev_portfolio = ons_train.prev_portfolio.copy()

        val_wealth = 1.0
        val_daily_wealth = [val_wealth]
        val_costs, val_turnovers = [], []
        val_no_trades = 0
        val_portfolios = []

        for day_idx, price_rel in enumerate(self.val_data):
            if hasattr(cost_model_val, 'update_state'):
                cost_model_val.update_state(
                    wealth_fraction=val_wealth,
                    stock_prices=self.val_prices[day_idx],
                )

            current_portfolio = ons_val.get_portfolio()
            val_portfolios.append(current_portfolio)
            prev_portfolio = ons_val.prev_portfolio.copy()

            tc = cost_model_val.compute_cost(prev_portfolio, current_portfolio)
            val_wealth *= (1 - tc)
            if tc == 0:
                val_no_trades += 1
            val_costs.append(tc)
            val_turnovers.append(np.sum(np.abs(current_portfolio - prev_portfolio)))

            val_wealth *= np.dot(current_portfolio, price_rel)
            val_daily_wealth.append(val_wealth)
            ons_val.update(price_rel, current_portfolio)

        result = {
            'parameters': {'delta': delta, 'cost_penalty': cost_penalty, 'alpha': alpha},
            'training': {
                'final_wealth':           train_results['final_wealth'],
                'daily_wealth':           train_results['daily_wealth'].tolist(),
                'portfolios_used':        train_results['portfolios_used'].tolist(),
                'transaction_costs':      train_results['transaction_costs'].tolist(),
                'total_transaction_cost': train_results['total_transaction_cost'],
                'avg_turnover':           train_results['avg_turnover'],
                'trade_frequency':        train_results['trade_frequency'],
                'num_days':               train_results['num_days'],
            },
            'validation': {
                'final_wealth':           val_wealth,
                'daily_wealth':           val_daily_wealth,
                'portfolios_used':        [p.tolist() for p in val_portfolios],
                'transaction_costs':      val_costs,
                'total_transaction_cost': sum(val_costs),
                'avg_turnover':           float(np.mean(val_turnovers)),
                'trade_frequency':        1 - (val_no_trades / len(self.val_data)),
                'num_days':               len(self.val_data),
            },
        }
        return result

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------
    def _optuna_objective(self, trial, search_space):
        """Called by Optuna for each trial."""
        params = {}
        for name, spec in search_space.items():
            kind = spec['type']
            if kind == 'categorical':
                params[name] = trial.suggest_categorical(name, spec['choices'])
            elif kind == 'float':
                params[name] = trial.suggest_float(name, spec['low'], spec['high'],
                                                    log=spec.get('log', False))
            elif kind == 'int':
                params[name] = trial.suggest_int(name, spec['low'], spec['high'],
                                                  log=spec.get('log', False))
            else:
                raise ValueError(f"Unknown search space type: {kind}")

        result = self.run_single_training(
            delta=params['delta'],
            cost_penalty=params['cost_penalty'],
            alpha=params['alpha'],
        )

        # These show up as extra columns in the Optuna dashboard
        trial.set_user_attr('train_final_wealth', result['training']['final_wealth'])
        trial.set_user_attr('val_final_wealth',   result['validation']['final_wealth'])
        trial.set_user_attr('val_total_cost',     result['validation']['total_transaction_cost'])
        trial.set_user_attr('val_trade_freq',     result['validation']['trade_frequency'])
        trial.set_user_attr('val_avg_turnover',   result['validation']['avg_turnover'])

        # Store for compatibility with save_experiment_results()
        self.all_results.append(result)

        return result['validation']['final_wealth']

    # ------------------------------------------------------------------
    # Main search entry-point
    # ------------------------------------------------------------------
    def optuna_search(self, search_space, n_trials=50, n_jobs=1,
                      storage=None, sampler=None, pruner=None, verbose=True):
        """
        Run Optuna HPO.

        Args:
            search_space: dict describing parameters, e.g.:
                {
                  'delta':        {'type': 'float', 'low': 0.05, 'high': 0.99},
                  'cost_penalty': {'type': 'float', 'low': 0,    'high': 2000},
                  'alpha':        {'type': 'float', 'low': 0.0,  'high': 0.1},
                }
            n_trials:  Total number of trials to run.
            n_jobs:    Parallel trials. Keep low (4-8) since each trial is CPU-heavy.
            storage:   Optuna storage backend — pass return value of make_optuna_storage().
            sampler:   Optuna sampler (default: TPESampler).
            pruner:    Optuna pruner (default: MedianPruner).
            verbose:   Print progress.
        """
        if sampler is None:
            sampler = TPESampler(seed=42, n_startup_trials=10)
        if pruner is None:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=self.model_name,
            storage=storage,
            load_if_exists=True,    # safe to restart — resumes from where it left off
        )

        if verbose:
            print(f"Optuna Search for: {self.model_name}")
            print(f"Sampler  : {type(sampler).__name__}")
            print(f"Trials   : {n_trials}  |  n_jobs: {n_jobs}")
            if storage is not None:
                print(f"Storage  : {storage}")
                print(f"Dashboard: run  `optuna-dashboard {storage}`  in another terminal")
            print(f"Search space:")
            for k, v in search_space.items():
                print(f"  {k}: {v}")
            print()

        self.study.optimize(
            lambda trial: self._optuna_objective(trial, search_space),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=verbose,
        )

        best_trial = self.study.best_trial
        self.best_params = {
            'delta':        best_trial.params['delta'],
            'cost_penalty': best_trial.params['cost_penalty'],
            'alpha':        best_trial.params['alpha'],
        }

        if verbose:
            print(f"\nBest trial #{best_trial.number}  (val wealth: {best_trial.value:.6f})")
            for k, v in self.best_params.items():
                print(f"  {k}: {v}")

        return self.best_params

    # ------------------------------------------------------------------
    # run_test — identical to original
    # ------------------------------------------------------------------
    def run_test(self, best_params, verbose=True):
        """Retrain on train+val with best hyperparams, then test."""
        from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"RETRAINING ON TRAIN+VAL, THEN TESTING")
            print(f"{'=' * 80}")

        combined_data   = np.vstack([self.train_data, self.val_data])
        combined_prices = (np.vstack([self.train_prices, self.val_prices])
                           if self.train_prices is not None else None)

        cost_model_retrain = copy.deepcopy(self.cost_model)
        ons_retrain = OnlineNewtonStepCosts(
            n_stocks=self.n_stocks, eta=self.eta, beta=self.beta,
            delta=best_params['delta'], cost_model=cost_model_retrain,
            cost_penalty=best_params['cost_penalty'], alpha=best_params['alpha'],
        )
        retrain_results = ons_retrain.simulate_trading(
            combined_data, stock_prices_sequence=combined_prices, verbose=False
        )
        self.retrain_results = {
            'final_wealth':           retrain_results['final_wealth'],
            'daily_wealth':           retrain_results['daily_wealth'],
            'portfolios_used':        retrain_results['portfolios_used'],
            'transaction_costs':      retrain_results['transaction_costs'],
            'turnovers':              retrain_results['turnovers'],
            'total_transaction_cost': retrain_results['total_transaction_cost'],
            'avg_turnover':           retrain_results['avg_turnover'],
            'trade_frequency':        retrain_results['trade_frequency'],
            'num_days':               retrain_results['num_days'],
        }

        if verbose:
            print(f"Retraining complete: wealth={retrain_results['final_wealth']:.4f}, "
                  f"trade_freq={retrain_results['trade_frequency']:.2%}\n"
                  f"Testing on {len(self.test_data)} days...")

        cost_model_test = copy.deepcopy(self.cost_model)
        ons_test = OnlineNewtonStepCosts(
            n_stocks=self.n_stocks, eta=self.eta, beta=self.beta,
            delta=best_params['delta'], cost_model=cost_model_test,
            cost_penalty=best_params['cost_penalty'], alpha=best_params['alpha'],
        )
        ons_test.A              = ons_retrain.A.copy()
        ons_test.b              = ons_retrain.b.copy()
        ons_test.p_t            = ons_retrain.p_t.copy()
        ons_test.prev_portfolio = ons_retrain.prev_portfolio.copy()

        test_wealth = 1.0
        test_daily_wealth = [test_wealth]
        test_costs, test_costs_dollars, test_turnovers = [], [], []
        test_no_trades = 0
        test_portfolios, test_traded_flags = [], []

        for day_idx, price_rel in enumerate(self.test_data):
            if hasattr(cost_model_test, 'update_state'):
                cost_model_test.update_state(
                    wealth_fraction=test_wealth,
                    stock_prices=self.test_prices[day_idx],
                )
            current_portfolio = ons_test.get_portfolio()
            prev_portfolio    = ons_test.prev_portfolio.copy()
            test_portfolios.append(current_portfolio.copy())

            tc          = cost_model_test.compute_cost(prev_portfolio, current_portfolio)
            dollar_cost = test_wealth * tc * self.initial_capital
            test_wealth *= (1 - tc)

            traded = dollar_cost > 0
            test_traded_flags.append(traded)
            if not traded:
                test_no_trades += 1

            test_costs.append(tc)
            test_costs_dollars.append(dollar_cost)
            test_turnovers.append(np.sum(np.abs(current_portfolio - prev_portfolio)))

            test_wealth *= np.dot(current_portfolio, price_rel)
            test_daily_wealth.append(test_wealth)
            ons_test.update(price_rel, current_portfolio)

        self.test_results = {
            'final_wealth':              test_wealth,
            'daily_wealth':              test_daily_wealth,
            'portfolios_used':           test_portfolios,
            'transaction_costs':         test_costs,
            'transaction_costs_dollars': test_costs_dollars,
            'turnovers':                 test_turnovers,
            'total_transaction_cost':    sum(test_costs),
            'avg_turnover':              float(np.mean(test_turnovers)),
            'trade_frequency':           1 - (test_no_trades / len(self.test_data)),
            'traded_flags':              test_traded_flags,
            'num_days':                  len(self.test_data),
            'net_return_pct':            (test_wealth - 1.0) * 100,
            'cost_drag_pct':             sum(test_costs) * 100,
            'ons_initial_portfolio':     ons_test.prev_portfolio.tolist(),
        }

        if verbose:
            print(f"\nTEST RESULTS:")
            print(f"  Final Wealth : {test_wealth:.4f}  ({self.test_results['net_return_pct']:.2f}%)")
            print(f"  Total Cost   : {self.test_results['cost_drag_pct']:.4f}% drag")
            print(f"  Trade Freq   : {self.test_results['trade_frequency']:.2%}")
            print(f"  Avg Turnover : {float(np.mean(test_turnovers)):.4f}")
            print(f"{'=' * 80}\n")

        return self.test_results

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    def print_optuna_summary(self, top_n=10):
        if self.study is None:
            print("No study found. Run optuna_search() first.")
            return
        trials_df  = self.study.trials_dataframe()
        param_cols = [c for c in trials_df.columns if c.startswith('params_')]
        print(f"\nTop {top_n} trials by validation wealth:")
        print(
            trials_df
            .sort_values('value', ascending=False)
            .head(top_n)[['number', 'value'] + param_cols]
            .to_string(index=False)
        )


# ======================================================================
# Helper: pick the right storage backend based on n_jobs
# ======================================================================
def make_optuna_storage(db_path="optuna_study.db", n_jobs=1):
    """
    Always uses SQLite — works fine for up to ~8 parallel workers.
    No extra setup needed (SQLite is built into Python).

    Dashboard command (run in a SEPARATE terminal while the study is running):
        optuna-dashboard sqlite:///ons_hpo.db
    Then open http://localhost:8080
    """
    storage_url = f"sqlite:///{db_path}"
    print(f"[Storage]   SQLite → {db_path}")
    print(f"[Dashboard] Open a new terminal and run:")
    print(f"            optuna-dashboard sqlite:///{db_path}")
    print(f"            Then open: http://localhost:8080")
    return storage_url


# ======================================================================
# Drop-in replacement for run_3split_experiment
# ======================================================================
def run_3split_experiment_optuna(
        stocks, train_start_date, train_end_date, val_end_date, test_end_date,
        cost_models, search_spaces, results_base_dir,
        n_trials=50, n_jobs=1, save_results=True, initial_capital=1000,
        db_path="ons_hpo.db",
):
    """
    Same interface as run_3split_experiment() but uses Optuna instead of grid search.

    Extra args vs original:
        search_spaces : replaces parameter_grid  (see examples below)
        n_trials      : replaces exhaustive grid
        db_path       : SQLite / journal file for the dashboard

    search_spaces example — continuous (recommended, explores more territory):
        {
          'IB Fixed ($10k)': {
              'delta':        {'type': 'float', 'low': 0.05,  'high': 0.99},
              'cost_penalty': {'type': 'float', 'low': 0,     'high': 2000},
              'alpha':        {'type': 'float', 'low': 0.0,   'high': 0.1},
          }
        }

    search_spaces example — categorical (mirrors original grid):
        {
          'IB Fixed ($10k)': {
              'delta':        {'type': 'categorical', 'choices': [0.125, 0.5, 0.8, 0.9]},
              'cost_penalty': {'type': 'categorical', 'choices': [0, 10, 1000]},
              'alpha':        {'type': 'categorical', 'choices': [0.0, 0.01, 0.05]},
          }
        }
    """
    from utils.data_prep import prepare_stock_data_3split
    from benchmarks.portfolio.buy_and_hold import run_buy_and_hold

    print("Preparing data...")
    data_dict = prepare_stock_data_3split(
        stocks=stocks,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        test_end_date=test_end_date,
    )

    train_data   = data_dict['train_price_relatives']
    val_data     = data_dict['val_price_relatives']
    test_data    = data_dict['test_price_relatives']
    train_prices = data_dict['train_actual_prices']
    val_prices   = data_dict['val_actual_prices']
    test_prices  = data_dict['test_actual_prices']
    stock_names  = data_dict['stock_names']

    print("Computing benchmarks...")
    bah_uniform_results = run_buy_and_hold(test_data)
    nasdaq_test_results = sp500_test_results = None
    if 'benchmark_test_relatives' in data_dict:
        if data_dict['benchmark_test_relatives'].get('NASDAQ') is not None:
            nasdaq_test_results = run_buy_and_hold(data_dict['benchmark_test_relatives']['NASDAQ'])
        if data_dict['benchmark_test_relatives'].get('SP500') is not None:
            sp500_test_results  = run_buy_and_hold(data_dict['benchmark_test_relatives']['SP500'])

    if save_results:
        timestamp   = datetime.now().strftime('%d-%m-%y_%H-%M-%S')
        results_dir = Path(results_base_dir) / f"optuna_stocks{len(stocks)}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nResults will be saved to: {results_dir}\n")

    # One storage shared across all cost models in this run
    storage = make_optuna_storage(db_path=db_path, n_jobs=n_jobs)

    all_experiments = {}

    for model_name, cost_model in cost_models.items():
        print(f"\n{'=' * 80}\nCost Model: {model_name}\n{'=' * 80}\n")

        experiment = CostModelExperiment3SplitOptuna(
            cost_model=cost_model, model_name=model_name, stocks=stock_names,
            train_data=train_data, val_data=val_data, test_data=test_data,
            train_prices=train_prices, val_prices=val_prices, test_prices=test_prices,
            initial_capital=initial_capital,
        )

        best_params = experiment.optuna_search(
            search_space=search_spaces[model_name],
            n_trials=n_trials,
            n_jobs=n_jobs,
            storage=storage,
            verbose=True,
        )

        experiment.print_optuna_summary(top_n=10)
        experiment.run_test(best_params, verbose=True)
        all_experiments[model_name] = experiment

    # ── ONS-initial BAH benchmark ───────────────────────────────────────
    print("Computing ONS initial portfolio Buy-and-Hold benchmark...")
    bah_ons_initial_results = {}
    for model_name, experiment in all_experiments.items():
        ons_initial    = np.array(experiment.test_results['ons_initial_portfolio'])
        bah_ons_result = run_buy_and_hold(test_data, initial_weights=ons_initial)
        bah_ons_initial_results[model_name] = bah_ons_result
        print(f"  {model_name}: BAH(ONS init) = {bah_ons_result['final_wealth']:.4f}")

    # ── Save results ────────────────────────────────────────────────────
    if save_results:
        # from experiment_3split import NumpyEncoder, save_experiment_results

        metadata = {
            'run_info': {
                'timestamp':  timestamp,
                'date':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'hpo_method': 'optuna',
                'n_trials':   n_trials,
                'db_path':    str(db_path),
            },
            'experiment_settings': {
                'stocks':          stocks,
                'n_stocks':        len(stocks),
                'train_start':     train_start_date,
                'train_end':       train_end_date,
                'val_end':         val_end_date,
                'test_end':        test_end_date,
                'train_days':      len(train_data),
                'val_days':        len(val_data),
                'test_days':       len(test_data),
                'initial_capital': initial_capital,
                'n_jobs':          n_jobs,
            },
            'search_spaces': search_spaces,
            'cost_models':   list(cost_models.keys()),
            'benchmarks': {
                'buy_and_hold_uniform': {
                    'final_wealth': float(bah_uniform_results['final_wealth']),
                    'daily_wealth': bah_uniform_results['daily_wealth'].tolist(),
                },
                'buy_and_hold_ons_initial': {
                    mn: {
                        'initial_portfolio': bah_ons_initial_results[mn]['initial_weights'].tolist(),
                        'final_wealth':      float(bah_ons_initial_results[mn]['final_wealth']),
                        'daily_wealth':      bah_ons_initial_results[mn]['daily_wealth'].tolist(),
                    }
                    for mn in all_experiments
                },
                'nasdaq': {
                    'available':    nasdaq_test_results is not None,
                    'final_wealth': float(nasdaq_test_results['final_wealth']) if nasdaq_test_results else None,
                    'daily_wealth': nasdaq_test_results['daily_wealth'].tolist() if nasdaq_test_results else None,
                },
                'sp500': {
                    'available':    sp500_test_results is not None,
                    'final_wealth': float(sp500_test_results['final_wealth']) if sp500_test_results else None,
                    'daily_wealth': sp500_test_results['daily_wealth'].tolist() if sp500_test_results else None,
                },
            },
        }
        save_experiment_results(results_dir, all_experiments, metadata)

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 80}\nFINAL TEST RESULTS SUMMARY\n{'=' * 80}")
    print(f"\nBENCHMARKS:")
    print(f"  Uniform BAH : {bah_uniform_results['final_wealth']:.4f}")
    for mn in all_experiments:
        print(f"  BAH (ONS init): {bah_ons_initial_results[mn]['final_wealth']:.4f}")
    if nasdaq_test_results:
        print(f"  NASDAQ  : {nasdaq_test_results['final_wealth']:.4f}")
    if sp500_test_results:
        print(f"  S&P 500 : {sp500_test_results['final_wealth']:.4f}")

    print(f"\nONS RESULTS:")
    for model_name, experiment in all_experiments.items():
        print(f"\n{model_name}:")
        for k, v in experiment.best_params.items():
            print(f"  {k}: {v}")
        tr = experiment.test_results
        print(f"  Final Wealth : {tr['final_wealth']:.4f}  ({tr['net_return_pct']:.2f}%)")
        print(f"  Cost Drag    : {tr['cost_drag_pct']:.4f}%")
        print(f"  Trade Freq   : {tr['trade_frequency']:.2%}")

    return all_experiments


# ======================================================================
# __main__
# ======================================================================
if __name__ == "__main__":

    TRAIN_START_DATE = "2021-02-01"
    TRAIN_END_DATE   = "2023-02-01"
    VAL_END_DATE     = "2024-02-01"
    TEST_END_DATE    = "2026-02-01"

    SAVE_RESULTS     = True
    N_TRIALS         = 40
    N_JOBS           = 3       # storage backend is chosen automatically based on this
    INITIAL_CAPITAL  = 10000
    RESULTS_BASE_DIR = '/home/robert/PycharmProjects/OnlinePortfolioSelection/results/portfolio/WithCosts/ONS'

    # ── Dashboard ─────────────────────────────────────────────────────
    # While this script is running, open a NEW terminal and run ONE of:
    #   N_JOBS == 1:  optuna-dashboard sqlite:///ons_hpo.db
    #   N_JOBS  > 1:  optuna-dashboard journal:///ons_hpo_journal.log
    # Then open http://localhost:8080 in your browser.
    DB_PATH = "ons_hpo.db"

    from algorithms.transactions.interactive_brokers import InteractiveBrokersCostUSD

    price_to_k = INITIAL_CAPITAL / 1000

    experiment_configs = [
        {
            'name':   'High Growth Tech (5 stocks)',
            'stocks': ["NVDA", "TSLA", "AMD", "PLTR", "SNOW"],
        },
    ]

    for i, config in enumerate(experiment_configs, 1):
        print(f"\n{'=' * 100}")
        print(f"EXPERIMENT {i}/{len(experiment_configs)}: {config['name']}")
        print('=' * 100)

        cost_models = {
            f"IB Fixed (${price_to_k}k)": InteractiveBrokersCostUSD(
                initial_capital=INITIAL_CAPITAL, pricing_type="fixed"
            ),
        }

        # ── Search space ──────────────────────────────────────────────
        # Option A: continuous (recommended — lets TPE explore freely)
        search_spaces = {
            f"IB Fixed (${price_to_k}k)": {
                'delta':        {'type': 'float', 'low': 0.05,  'high': 0.99},
                'cost_penalty': {'type': 'float', 'low': 0.0,   'high': 2000.0},
                'alpha':        {'type': 'float', 'low': 0.0,   'high': 0.15},
            }
        }

        # Option B: categorical (exact replica of original grid)
        # search_spaces = {
        #     f"IB Fixed (${price_to_k}k)": {
        #         'delta':        {'type': 'categorical', 'choices': [0.125, 0.5, 0.8, 0.9]},
        #         'cost_penalty': {'type': 'categorical', 'choices': [0, 10, 1000]},
        #         'alpha':        {'type': 'categorical', 'choices': [0.0, 0.01, 0.05]},
        #     }
        # }

        run_3split_experiment_optuna(
            stocks=config['stocks'],
            train_start_date=TRAIN_START_DATE,
            train_end_date=TRAIN_END_DATE,
            val_end_date=VAL_END_DATE,
            test_end_date=TEST_END_DATE,
            cost_models=cost_models,
            search_spaces=search_spaces,
            results_base_dir=RESULTS_BASE_DIR,
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            save_results=SAVE_RESULTS,
            initial_capital=INITIAL_CAPITAL,
            db_path=DB_PATH,
        )