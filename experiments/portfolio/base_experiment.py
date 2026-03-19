import copy
from abc import ABC, abstractmethod
from pathlib import Path
from joblib import Parallel, delayed
import json
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def _compute_risk_metrics(daily_wealth):
    """
    Compute risk-adjusted performance metrics from a daily wealth series.
    Returns a dict with sharpe, sortino, max_drawdown, calmar, annualized_return.
    """
    daily_wealth = np.array(daily_wealth)
    daily_returns = np.diff(daily_wealth) / daily_wealth[:-1]
    n_days = len(daily_returns)

    # Sharpe Ratio (annualized)
    if n_days > 1 and np.std(daily_returns) > 1e-10:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino Ratio (annualized, using downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        sortino = (np.mean(daily_returns) / downside_std) * np.sqrt(252) if downside_std > 1e-10 else 0.0
    else:
        sortino = 0.0

    # Max Drawdown
    cumulative = daily_wealth / daily_wealth[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_drawdown = float(np.max(drawdowns))

    # Annualized Return
    annualized_return = (daily_wealth[-1] / daily_wealth[0]) ** (252 / n_days) - 1 if n_days > 0 else 0.0

    # Calmar Ratio (annualized return / max drawdown)
    calmar = annualized_return / max_drawdown if max_drawdown > 1e-10 else 0.0

    return {
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_drawdown': round(float(max_drawdown), 4),
        'calmar': round(float(calmar), 4),
        'annualized_return': round(float(annualized_return), 4),
    }


class _ExperimentMixin:
    """
    Provides the shared validation and test loops.
    Subclasses must set: n_stocks, val_data, val_prices, test_data, test_prices, cost_model, initial_capital
    """
    def _run_val_loop(self, algo_val, cost_model_val):
        """
        Run the validation loop on self.val_data.
        Returns a dict with val metrics.
        """
        val_wealth = 1.0
        val_daily_wealth = [val_wealth]
        val_costs, val_costs_dollars, val_turnovers = [], [], []
        val_no_trades = 0
        val_portfolios, val_traded_flags = [], []

        for day_idx, price_rel in enumerate(self.val_data):

            cost_model_val.update_state(wealth_fraction=val_wealth, stock_prices=self.val_prices[day_idx])

            current_portfolio = algo_val.get_portfolio()
            val_portfolios.append(current_portfolio)
            prev_portfolio = algo_val.prev_portfolio.copy()

            tc = cost_model_val.compute_cost(prev_portfolio, current_portfolio)
            dollar_cost = val_wealth * tc * self.initial_capital
            val_wealth *= (1 - tc)

            traded = dollar_cost > 0
            val_traded_flags.append(traded)
            if not traded:
                val_no_trades += 1

            val_costs.append(tc)
            val_costs_dollars.append(dollar_cost)
            val_turnovers.append(np.sum(np.abs(current_portfolio - prev_portfolio)))

            val_wealth *= np.dot(current_portfolio, price_rel)
            val_daily_wealth.append(val_wealth)
            self._algo_update(algo_val, price_rel, current_portfolio)

        risk_metrics = _compute_risk_metrics(val_daily_wealth)

        return {
            'final_wealth': val_wealth,
            'daily_wealth': val_daily_wealth,
            'portfolios_used': [p.tolist() for p in val_portfolios],
            'transaction_costs': val_costs,
            'transaction_costs_dollars': val_costs_dollars,
            'turnovers': val_turnovers,
            'total_transaction_cost': sum(val_costs),
            'avg_turnover': float(np.mean(val_turnovers)),
            'trade_frequency': 1 - (val_no_trades / len(self.val_data)),
            'traded_flags': val_traded_flags,
            'num_days': len(self.val_data),
            'net_return_pct': (val_wealth - 1.0) * 100,
            'cost_drag_pct': sum(val_costs) * 100,
            **risk_metrics,
        }

    def _run_test_loop(self, algo_test, cost_model_test):
        """
        Run the test loop on self.test_data.
        Returns a dict with test metrics.
        """
        test_wealth = 1.0
        test_daily_wealth = [test_wealth]
        test_costs, test_costs_dollars, test_turnovers = [], [], []
        test_no_trades = 0
        test_portfolios, test_traded_flags = [], []

        for day_idx, price_rel in enumerate(self.test_data):
            cost_model_test.update_state(wealth_fraction=test_wealth, stock_prices=self.test_prices[day_idx],)

            current_portfolio = algo_test.get_portfolio()
            prev_portfolio = algo_test.prev_portfolio.copy()
            test_portfolios.append(current_portfolio.copy())

            tc = cost_model_test.compute_cost(prev_portfolio, current_portfolio)
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
            self._algo_update(algo_test, price_rel, current_portfolio)

        risk_metrics = _compute_risk_metrics(test_daily_wealth)

        return {
            'final_wealth': test_wealth,
            'daily_wealth': test_daily_wealth,
            'portfolios_used': test_portfolios,
            'transaction_costs': test_costs,
            'transaction_costs_dollars': test_costs_dollars,
            'turnovers': test_turnovers,
            'total_transaction_cost': sum(test_costs),
            'avg_turnover': float(np.mean(test_turnovers)),
            'trade_frequency': 1 - (test_no_trades / len(self.test_data)),
            'traded_flags': test_traded_flags,
            'num_days': len(self.test_data),
            'net_return_pct': (test_wealth - 1.0) * 100,
            'cost_drag_pct': sum(test_costs) * 100,
            **risk_metrics,
        }

    def _algo_update(self, algo, price_rel, portfolio_used):
        """
        Default update call — works for ONS-style algorithms.
        Override in subclass for algorithms with different update signatures.
        """
        algo.update(price_rel, portfolio_used)


class BaseGridExperiment(_ExperimentMixin, ABC):
    """
    Abstract base for 3-split experiments using joblib parallel grid search.
    """

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

        self.all_results = []
        self.best_params = None
        self.retrain_results = None
        self.test_results = None

    @abstractmethod
    def _build_algorithm(self, params: dict, cost_model):
        """Build and return a fresh algorithm instance for the given params."""

    @abstractmethod
    def _transfer_state(self, algo_src, algo_dst):
        """Copy learned state from algo_src (trained) into algo_dst (val/test)."""

    @abstractmethod
    def _param_combinations(self, parameter_grid: dict) -> list:
        """Return list of param dicts from the grid."""

    @property
    @abstractmethod
    def initial_portfolio_key(self) -> str:
        """Key used in test_results to store the initial portfolio for BAH."""

    def run_single_training(self, params: dict) -> dict:
        cost_model_train = copy.deepcopy(self.cost_model)
        algo_train = self._build_algorithm(params, cost_model_train)
        train_results = algo_train.simulate_trading(
            self.train_data, stock_prices_sequence=self.train_prices,
            verbose=False,
        )

        cost_model_val = copy.deepcopy(self.cost_model)
        algo_val = self._build_algorithm(params, cost_model_val)
        self._transfer_state(algo_train, algo_val)
        val_metrics = self._run_val_loop(algo_val, cost_model_val)

        return {
            'parameters': params,
            'training': {
                'final_wealth': train_results['final_wealth'],
                'daily_wealth': train_results['daily_wealth'].tolist(),
                'portfolios_used': train_results['portfolios_used'].tolist(),
                'transaction_costs': train_results['transaction_costs'].tolist(),
                'total_transaction_cost': train_results['total_transaction_cost'],
                'avg_turnover': train_results['avg_turnover'],
                'trade_frequency': train_results['trade_frequency'],
                'num_days': train_results['num_days'],
            },
            'validation': val_metrics,
        }

    def grid_search(self, parameter_grid: dict, verbose=True, n_jobs=-1) -> dict:
        combos = self._param_combinations(parameter_grid)

        if verbose:
            print(f"Grid Search for {self.model_name}")
            print(f"Testing {len(combos)} combinations, n_jobs={n_jobs}\n")

        all_results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(self.run_single_training)(p) for p in combos)
        self.all_results = all_results

        best = max(all_results, key=lambda x: x['validation']['final_wealth']) # Choosing the best parameter based on final wealth, MAY BE CHANGED IF FINAL PARAMS ARE DECIDED TO BE CHOSEN IN ANOTHER WAY
        self.best_params = best['parameters']

        if verbose:
            print(f"\nCompleted {len(all_results)} experiments")
            print(f"Best (on validation):")
            for k, v in self.best_params.items():
                print(f"{k}: {v}")
            print(f"Train wealth: {best['training']['final_wealth']:.4f}")
            print(f"Val wealth: {best['validation']['final_wealth']:.4f}")

        return self.best_params

    def run_test(self, best_params: dict, verbose=True) -> dict:
        if verbose:
            print(f"Retraining on Train+Val")

        combined_data = np.vstack([self.train_data, self.val_data])
        combined_prices = (np.vstack([self.train_prices, self.val_prices]) if self.train_prices is not None else None)

        cost_model_retrain = copy.deepcopy(self.cost_model)
        algo_retrain = self._build_algorithm(best_params, cost_model_retrain)
        retrain_results = algo_retrain.simulate_trading(combined_data, stock_prices_sequence=combined_prices, verbose=False)

        self.retrain_results = {
            'final_wealth': retrain_results['final_wealth'],
            'daily_wealth': retrain_results['daily_wealth'],
            'portfolios_used': retrain_results['portfolios_used'],
            'transaction_costs': retrain_results['transaction_costs'],
            'turnovers': retrain_results['turnovers'],
            'total_transaction_cost': retrain_results['total_transaction_cost'],
            'avg_turnover': retrain_results['avg_turnover'],
            'trade_frequency': retrain_results['trade_frequency'],
            'num_days': retrain_results['num_days'],
        }

        if verbose:
            print(f"Retraining: wealth={retrain_results['final_wealth']:.4f}, "
                  f"Testing on {len(self.test_data)} days")

        cost_model_test = copy.deepcopy(self.cost_model)
        algo_test = self._build_algorithm(best_params, cost_model_test)
        self._transfer_state(algo_retrain, algo_test)
        test_metrics = self._run_test_loop(algo_test, cost_model_test)
        test_metrics[self.initial_portfolio_key] = algo_test.prev_portfolio.tolist()
        self.test_results = test_metrics

        if verbose:
            print(f"\nTEST RESULTS:")
            print(f"Final Wealth: {test_metrics['final_wealth']:.4f} "
                  f"({test_metrics['net_return_pct']:.2f}%)")
            print(f"Cost Drag: {test_metrics['cost_drag_pct']:.4f}%")
            print(f"Trade Freq: {test_metrics['trade_frequency']:.2%}")
            print(f"Sharpe: {test_metrics['sharpe']:.4f}")
            print(f"Sortino: {test_metrics['sortino']:.4f}")
            print(f"Max Drawdown: {test_metrics['max_drawdown']:.4f}")
            print(f"Calmar: {test_metrics['calmar']:.4f}")

        return self.test_results

    def print_summary(self, top_n=10):
        if not self.all_results:
            print("No results yet.")
            return
        rows = [{'val_wealth': r['validation']['final_wealth'], **r['parameters']} for r in self.all_results]
        df = pd.DataFrame(rows).sort_values('val_wealth', ascending=False)
        print(f"\nTop {top_n} grid combinations:")
        print(df.head(top_n).to_string(index=False))


class BaseOptunaExperiment(_ExperimentMixin, ABC):
    """
    Abstract base for 3-split experiments using Optuna hyperparameter optimization.
    """

    def __init__(self, cost_model, model_name, stocks, train_data, val_data, test_data, train_prices=None,
                 val_prices=None, test_prices=None, initial_capital=None):
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

        self.study = None
        self.best_params = None
        self.all_results = []
        self.retrain_results = None
        self.test_results = None

    @abstractmethod
    def _build_algorithm(self, params: dict, cost_model):
        """Build and return a fresh algorithm instance."""

    @abstractmethod
    def _transfer_state(self, algo_src, algo_dst):
        """Copy learned state from algo_src into algo_dst."""

    @abstractmethod
    def _suggest_params(self, trial, search_space: dict) -> dict:
        """Use trial.suggest_* to draw parameters. Return a params dict."""

    @property
    @abstractmethod
    def initial_portfolio_key(self) -> str:
        """Key used in test_results to store the initial portfolio for BAH."""

    def _default_suggest_params(self, trial, search_space: dict) -> dict:
        """Generic suggestion for float/int/categorical specs."""
        params = {}
        for name, spec in search_space.items():
            kind = spec['type']
            if kind == 'categorical':
                params[name] = trial.suggest_categorical(name, spec['choices'])
            elif kind == 'float':
                params[name] = trial.suggest_float(
                    name, spec['low'], spec['high'], log=spec.get('log', False))
            elif kind == 'int':
                params[name] = trial.suggest_int(
                    name, spec['low'], spec['high'], log=spec.get('log', False))
            else:
                raise ValueError(f"Unknown search space type: {kind}")
        return params

    def run_single_training(self, params: dict) -> dict:
        cost_model_train = copy.deepcopy(self.cost_model)
        algo_train = self._build_algorithm(params, cost_model_train)
        train_results = algo_train.simulate_trading(self.train_data, stock_prices_sequence=self.train_prices, verbose=False,)

        cost_model_val = copy.deepcopy(self.cost_model)
        algo_val = self._build_algorithm(params, cost_model_val)
        self._transfer_state(algo_train, algo_val)
        val_metrics = self._run_val_loop(algo_val, cost_model_val)

        return {
            'parameters': params,
            'training': {
                'final_wealth': train_results['final_wealth'],
                'daily_wealth': train_results['daily_wealth'].tolist(),
                'portfolios_used': train_results['portfolios_used'].tolist(),
                'transaction_costs': train_results['transaction_costs'].tolist(),
                'total_transaction_cost': train_results['total_transaction_cost'],
                'avg_turnover': train_results['avg_turnover'],
                'trade_frequency': train_results['trade_frequency'],
                'num_days': train_results['num_days'],
            },
            'validation': val_metrics,
        }

    def _optuna_objective(self, trial, search_space: dict) -> float:
        params = self._suggest_params(trial, search_space)
        result = self.run_single_training(params)

        val = result['validation']

        trial.set_user_attr('train_final_wealth', result['training']['final_wealth'])
        trial.set_user_attr('val_final_wealth', val['final_wealth'])
        trial.set_user_attr('val_total_cost', val['total_transaction_cost'])
        trial.set_user_attr('val_trade_freq', val['trade_frequency'])
        trial.set_user_attr('val_avg_turnover', val['avg_turnover'])
        trial.set_user_attr('val_sharpe', val['sharpe'])
        trial.set_user_attr('val_sortino', val['sortino'])
        trial.set_user_attr('val_max_drawdown', val['max_drawdown'])
        trial.set_user_attr('val_calmar', val['calmar'])

        self.all_results.append(result)
        if val['final_wealth'] < 1.0:
            return -999.0

        return val['sortino']

    def optuna_search(self, search_space: dict, n_trials=50, n_jobs=1, storage=None, sampler=None, pruner=None, verbose=True) -> dict:
        if sampler is None:
            sampler = TPESampler(seed=42, n_startup_trials=10)
        if pruner is None:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        self.study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner,
                                         study_name=self.model_name, storage=storage, load_if_exists=True,)

        if verbose:
            print(f"Optuna Search for: {self.model_name}")
            print(f"Sampler: {type(sampler).__name__}")
            print(f"Trials: {n_trials} | n_jobs: {n_jobs}")
            if storage:
                print(f"Storage: {storage}")
                print(f"Dashboard: optuna-dashboard {storage}")
            print(f"Search space:")
            for k, v in search_space.items():
                print(f"{k}: {v}")
            print()

        self.study.optimize(
            lambda trial: self._optuna_objective(trial, search_space),
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=verbose,
        )

        best_trial = self.study.best_trial
        self.best_params = best_trial.params

        if verbose:
            print(f"\nBest trial #{best_trial.number} "
                  f"(val wealth: {best_trial.value:.6f})")
            for k, v in self.best_params.items():
                print(f"  {k}: {v}")

        return self.best_params

    def run_test(self, best_params: dict, verbose=True) -> dict:
        if verbose:
            print(f"Retraining on Train+Val")

        combined_data = np.vstack([self.train_data, self.val_data])
        combined_prices = (np.vstack([self.train_prices, self.val_prices]) if self.train_prices is not None else None)

        cost_model_retrain = copy.deepcopy(self.cost_model)
        algo_retrain = self._build_algorithm(best_params, cost_model_retrain)
        retrain_results = algo_retrain.simulate_trading(combined_data, stock_prices_sequence=combined_prices, verbose=False)

        self.retrain_results = {
            'final_wealth': retrain_results['final_wealth'],
            'daily_wealth': retrain_results['daily_wealth'],
            'portfolios_used': retrain_results['portfolios_used'],
            'transaction_costs': retrain_results['transaction_costs'],
            'turnovers': retrain_results['turnovers'],
            'total_transaction_cost': retrain_results['total_transaction_cost'],
            'avg_turnover': retrain_results['avg_turnover'],
            'trade_frequency': retrain_results['trade_frequency'],
            'num_days': retrain_results['num_days'],
        }

        if verbose:
            print(f"Retraining: wealth={retrain_results['final_wealth']:.4f}, "
                  f"Testing on {len(self.test_data)} days")

        cost_model_test = copy.deepcopy(self.cost_model)
        algo_test = self._build_algorithm(best_params, cost_model_test)
        self._transfer_state(algo_retrain, algo_test)
        test_metrics = self._run_test_loop(algo_test, cost_model_test)
        test_metrics[self.initial_portfolio_key] = algo_test.prev_portfolio.tolist()
        self.test_results = test_metrics

        if verbose:
            print(f"\nTEST RESULTS:")
            print(f"Final Wealth: {test_metrics['final_wealth']:.4f} "
                  f"({test_metrics['net_return_pct']:.2f}%)")
            print(f"Cost Drag: {test_metrics['cost_drag_pct']:.4f}%")
            print(f"Trade Freq: {test_metrics['trade_frequency']:.2%}")
            print(f"Sharpe: {test_metrics['sharpe']:.4f}")
            print(f"Sortino: {test_metrics['sortino']:.4f}")
            print(f"Max Drawdown: {test_metrics['max_drawdown']:.4f}")
            print(f"Calmar: {test_metrics['calmar']:.4f}")

        return self.test_results

    def print_optuna_summary(self, top_n=10):
        if self.study is None:
            print("No study. Run optuna_search() first.")
            return
        trials_df = self.study.trials_dataframe()
        param_cols = [c for c in trials_df.columns if c.startswith('params_')]
        print(f"\nTop {top_n} trials by validation wealth:")
        print(trials_df.sort_values('value', ascending=False).head(top_n)[['number', 'value'] + param_cols].to_string(index=False))

def make_optuna_storage(db_path="optuna_study.db"):
    """
    Returns a SQLite storage URL. Works with up to ~8 parallel workers.

    Run in a separate terminal: optuna-dashboard sqlite:///optuna_study.db
    Then open http://localhost:8080
    """
    storage_url = f"sqlite:///{db_path}"
    print(f"[Storage] SQLite -> {db_path}")
    print(f"[Dashboard] optuna-dashboard {storage_url}")
    print(f"http://localhost:8080")
    return storage_url

def save_experiment_results(results_dir, experiments: dict, data_dict: dict,
                            run_info: dict, bah_results: dict):
    """
    results_dir  : Path or str where files will be saved
    experiments  : {model_name: experiment_instance}
    data_dict    : output of prepare_stock_data_3split (for metadata)
    run_info     : dict with hpo_method, n_trials, db_path, n_jobs etc.
    bah_results  : {
        'uniform': bah_uniform result,
        'ons_initial': bah_ons result,   (or 'barrons_initial' etc.)
    }
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%d-%m-%y_%H-%M-%S')

    benchmarks_data = {
        name: {
            'final_wealth': float(result['final_wealth']),
            'daily_wealth': result['daily_wealth'].tolist()
                if isinstance(result['daily_wealth'], np.ndarray)
                else result['daily_wealth'],
        }
        for name, result in bah_results.items()
        if result is not None
    }

    with open(results_dir / 'benchmarks.json', 'w') as f:
        json.dump(benchmarks_data, f, indent=2, cls=NumpyEncoder)
    print(f"Benchmarks saved to: {results_dir / 'benchmarks.json'}")

    metadata = {
        'run_info': {
            'timestamp': timestamp,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **run_info,
        },
        'experiment_settings': {
            'stocks': data_dict['stock_names'],
            'n_stocks': len(data_dict['stock_names']),
            'has_cash': data_dict.get('has_cash', False),
            'cash_index': data_dict.get('cash_index', None),
            'train_days': len(data_dict['train_price_relatives']),
            'val_days': len(data_dict['val_price_relatives']),
            'test_days': len(data_dict['test_price_relatives']),
            'train_start': str(data_dict['train_val_split_date']),
            'val_end': str(data_dict['val_test_split_date']),
        },
        'benchmarks': {
            name: {'final_wealth': float(result['final_wealth'])}
            for name, result in bah_results.items()
            if result is not None
        },
    }

    with open(results_dir / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    print(f"Metadata saved to: {results_dir / 'experiment_metadata.json'}")

    # all_results.csv
    all_results_list = []
    for model_name, experiment in experiments.items():
        for result in experiment.all_results:
            row = {
                'model': model_name,
                **result['parameters'],
                'train_final_wealth': result['training']['final_wealth'],
                'train_total_cost': result['training']['total_transaction_cost'],
                'train_trade_freq': result['training']['trade_frequency'],
                'val_final_wealth': result['validation']['final_wealth'],
                'val_total_cost': result['validation']['total_transaction_cost'],
                'val_trade_freq': result['validation']['trade_frequency'],
            }
            all_results_list.append(row)

    pd.DataFrame(all_results_list).to_csv(results_dir / 'all_results.csv', index=False)
    print(f"All results saved to: {results_dir / 'all_results.csv'}")

    # detailed_results.json
    detailed_results = {}
    for model_name, experiment in experiments.items():
        detailed_results[model_name] = {
            'best_parameters': experiment.best_params,
            'all_parameter_combinations': experiment.all_results,
            'retrain_on_train_val': {
                'final_wealth': experiment.retrain_results['final_wealth'],
                'daily_wealth': experiment.retrain_results['daily_wealth'].tolist()
                    if isinstance(experiment.retrain_results['daily_wealth'], np.ndarray)
                    else experiment.retrain_results['daily_wealth'],
                'portfolios_used': experiment.retrain_results['portfolios_used'].tolist()
                    if isinstance(experiment.retrain_results['portfolios_used'], np.ndarray)
                    else experiment.retrain_results['portfolios_used'],
                'transaction_costs': experiment.retrain_results['transaction_costs'].tolist()
                    if isinstance(experiment.retrain_results['transaction_costs'], np.ndarray)
                    else experiment.retrain_results['transaction_costs'],
                'turnovers': experiment.retrain_results['turnovers'].tolist()
                    if isinstance(experiment.retrain_results['turnovers'], np.ndarray)
                    else experiment.retrain_results['turnovers'],
                'total_transaction_cost': experiment.retrain_results['total_transaction_cost'],
                'avg_turnover': experiment.retrain_results['avg_turnover'],
                'trade_frequency': experiment.retrain_results['trade_frequency'],
                'num_days': experiment.retrain_results['num_days'],
            },
            'test': {
                **experiment.test_results,
                'portfolios_used': [
                    p.tolist() if isinstance(p, np.ndarray) else p
                    for p in experiment.test_results['portfolios_used']
                ],
                'test_price_relatives': data_dict['test_price_relatives'].tolist()
            },
        }

    with open(results_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2, cls=NumpyEncoder)
    print(f"Detailed results saved to: {results_dir / 'detailed_results.json'}")
    print(f"\nAll results saved in: {results_dir}")

class NumpyEncoder(json.JSONEncoder):
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