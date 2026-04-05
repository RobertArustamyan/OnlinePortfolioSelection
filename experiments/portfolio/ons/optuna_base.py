"""
ons_optuna_experiment.py

Optuna HPO experiment for OnlineNewtonStepCosts.
"""
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
import yfinance as yf

from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts
from experiments.portfolio.base_experiment import BaseOptunaExperiment
from algorithms.transactions.interactive_brokers import InteractiveBrokersCostUSD
from utils.data_prep import prepare_stock_data_3split
from benchmarks.portfolio.buy_and_hold import run_buy_and_hold
from utils.io import make_optuna_storage, save_experiment_results, NumpyEncoder
from experiments.portfolio.ons.visualisation_ons import ExperimentPlotter
from experiments.portfolio.stability_analysis import run_stability_analysis


class ONSOptunaExperiment(BaseOptunaExperiment):
    """
    3-split Optuna HPO experiment for the ONS algorithm.
    """

    def __init__(self, cost_model, model_name, stocks,
                 train_data, val_data, test_data,
                 train_prices=None, val_prices=None, test_prices=None,
                 initial_capital=None,optimize_metric="sortino"):
        super().__init__(
            cost_model=cost_model, model_name=model_name, stocks=stocks,
            train_data=train_data, val_data=val_data, test_data=test_data,
            train_prices=train_prices, val_prices=val_prices, test_prices=test_prices,
            initial_capital=initial_capital,optimize_metric=optimize_metric,
        )
        T = len(train_data) + len(val_data) + len(test_data)
        n = self.n_stocks
        self.eta = (n ** 1.25) / (np.sqrt(T * np.log(n * T)))
        self.beta = 1.0 / (8 * (n ** 0.25) * np.sqrt(T * np.log(n * T)))

    @property
    def initial_portfolio_key(self) -> str:
        return 'ons_initial_portfolio'

    def _build_algorithm(self, params: dict, cost_model) -> OnlineNewtonStepCosts:
        return OnlineNewtonStepCosts(
            n_stocks=self.n_stocks,
            eta=self.eta,
            beta=self.beta,
            delta=params['delta'],
            cost_model=cost_model,
            cost_penalty=params['cost_penalty'],
            alpha=params['alpha'],
        )

    def _transfer_state(self, algo_src: OnlineNewtonStepCosts,
                        algo_dst: OnlineNewtonStepCosts):
        algo_dst.A = algo_src.A.copy()
        algo_dst.b = algo_src.b.copy()
        algo_dst.p_t = algo_src.p_t.copy()
        algo_dst.prev_portfolio = algo_src.prev_portfolio.copy()

    def _suggest_params(self, trial, search_space: dict) -> dict:
        return self._default_suggest_params(trial, search_space)

if __name__ == "__main__":
    load_dotenv()

    # Run mode
    RUN_MODE = "single"  # "single" or "stability"

    # Shared config
    STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    INITIAL_CAPITAL = 5000
    N_TRIALS = 8
    N_JOBS = 10
    CASH_POSITION = False

    search_space = {
        'delta': {'type': 'float', 'low': 0.05, 'high': 0.99},
        'cost_penalty': {'type': 'float', 'low': 0.0, 'high': 1.0},
        'alpha': {'type': 'float', 'low': 0.0, 'high': 0.15},
    }

    cost_model = InteractiveBrokersCostUSD(
        initial_capital=INITIAL_CAPITAL,
        pricing_type="fixed",
        cash_index=-1,
    )

    # Single run config
    TRAIN_START_DATE = "2021-02-01"
    TRAIN_END_DATE = "2023-02-01"
    VAL_END_DATE = "2024-02-01"
    TEST_END_DATE = "2026-02-01"
    SAVE_RESULTS = True
    INDEX_BENCHMARKS = True
    RESULTS_BASE_DIR = os.path.join(os.getenv("RESULT_DIR_ONS"), "Optuna")

    # Stability config
    STABILITY_WINDOWS = [
        ("2021-02-01", "2022-08-01", "2023-08-01", "2024-08-01"),
        ("2021-02-01", "2023-02-01", "2024-02-01", "2025-02-01"),
        ("2022-02-01", "2023-08-01", "2024-08-01", "2026-02-01"),
    ]

    if RUN_MODE == "single":
        price_to_k = INITIAL_CAPITAL / 1000
        sp500_prices = yf.download('^GSPC', start=TRAIN_START_DATE, end=TEST_END_DATE, auto_adjust=True)['Close'].squeeze()
        data_dict = prepare_stock_data_3split(
            stocks=STOCKS,
            train_start_date=TRAIN_START_DATE,
            train_end_date=TRAIN_END_DATE,
            val_end_date=VAL_END_DATE,
            test_end_date=TEST_END_DATE,
            include_index_benchmarks=INDEX_BENCHMARKS,
            include_cash=CASH_POSITION,
        )

        hyperparams_dir = os.getenv("HYPERPARAMS_ONS")
        db_path = os.path.join(hyperparams_dir, "ons.db")
        storage = make_optuna_storage(db_path=db_path)

        experiment = ONSOptunaExperiment(
            cost_model=cost_model,
            model_name=f"IB Fixed (${price_to_k}k)[AAPL MSFT GOOGL AMZN META]",
            stocks=data_dict['stock_names'],
            train_data=data_dict['train_price_relatives'],
            val_data=data_dict['val_price_relatives'],
            test_data=data_dict['test_price_relatives'],
            train_prices=data_dict['train_actual_prices'],
            val_prices=data_dict['val_actual_prices'],
            test_prices=data_dict['test_actual_prices'],
            initial_capital=INITIAL_CAPITAL,
            optimize_metric="sortino",
        )

        best_params = experiment.optuna_search(
            search_space=search_space,
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            storage=storage,
            verbose=True,
        )

        experiment.print_optuna_summary(top_n=10)
        experiment.run_test(best_params, verbose=True)

        experiment.regime_analysis(sp500_prices=sp500_prices, val_dates=data_dict['val_dates'], test_dates=data_dict['test_dates'], verbose=True,)

        bah_results = experiment.compute_benchmarks(
            test_price_relatives=data_dict['test_price_relatives'],
            benchmark_relatives=data_dict.get('benchmark_test_relatives', {}),
        )

        if SAVE_RESULTS:
            results_dir = (Path(RESULTS_BASE_DIR) / f"optuna_{len(STOCKS)}_{datetime.now().strftime('%d-%m-%y_%H-%M')}")
            save_experiment_results(
                results_dir=results_dir,
                experiments={experiment.model_name: experiment},
                data_dict=data_dict,
                run_info={'hpo_method': 'optuna', 'n_trials': N_TRIALS, 'n_jobs': N_JOBS, 'db_path': 'ons_hpo.db'},
                bah_results=bah_results,
            )
            plotter = ExperimentPlotter(results_dir)
            plotter.create_all_plots()

    elif RUN_MODE == "stability":
        run_stability_analysis(
            experiment_class=ONSOptunaExperiment,
            cost_model=cost_model,
            stocks=STOCKS,
            initial_capital=INITIAL_CAPITAL,
            optimize_metric="sortino",
            search_space=search_space,
            windows=STABILITY_WINDOWS,
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown RUN_MODE: '{RUN_MODE}'. Choose 'single' or 'stability'.")