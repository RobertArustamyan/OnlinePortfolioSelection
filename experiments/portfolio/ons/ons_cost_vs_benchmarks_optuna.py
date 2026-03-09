"""
ons_optuna_experiment.py

Optuna HPO experiment for OnlineNewtonStepCosts.
"""
from pathlib import Path
from datetime import datetime

import numpy as np
from algorithms.portfolio.online_newton_step_costs import OnlineNewtonStepCosts
from experiments.portfolio.base_experiment import BaseOptunaExperiment

from algorithms.transactions.interactive_brokers import InteractiveBrokersCostUSD
from utils.data_prep import prepare_stock_data_3split
from benchmarks.portfolio.buy_and_hold import run_buy_and_hold
from experiments.portfolio.base_experiment import make_optuna_storage
from experiments.portfolio.base_experiment import save_experiment_results, NumpyEncoder


class ONSOptunaExperiment(BaseOptunaExperiment):
    """
    3-split Optuna HPO experiment for the ONS algorithm.
    """

    def __init__(self, cost_model, model_name, stocks,
                 train_data, val_data, test_data,
                 train_prices=None, val_prices=None, test_prices=None,
                 initial_capital=None):
        super().__init__(
            cost_model=cost_model, model_name=model_name, stocks=stocks,
            train_data=train_data, val_data=val_data, test_data=test_data,
            train_prices=train_prices, val_prices=val_prices, test_prices=test_prices,
            initial_capital=initial_capital,
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


    TRAIN_START_DATE = "2021-02-01"
    TRAIN_END_DATE = "2023-02-01"
    VAL_END_DATE = "2024-02-01"
    TEST_END_DATE = "2026-02-01"

    # STOCKS = ["NVDA", "TSLA", "AMD", "PLTR", "SNOW"]
    STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  
    # STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "PLTR", "SNOW"]
    INITIAL_CAPITAL = 1000
    N_TRIALS = 120
    N_JOBS = 3
    SAVE_RESULTS = True
    RESULTS_BASE_DIR = '/home/robert/PycharmProjects/OnlinePortfolioSelection/results/portfolio/WithCosts/ONS'

    price_to_k = INITIAL_CAPITAL / 1000

    # Prepare data
    data_dict = prepare_stock_data_3split(
        stocks=STOCKS,
        train_start_date=TRAIN_START_DATE,
        train_end_date=TRAIN_END_DATE,
        val_end_date=VAL_END_DATE,
        test_end_date=TEST_END_DATE,
        include_cash=True
    )

    cost_model = InteractiveBrokersCostUSD(
        initial_capital=INITIAL_CAPITAL,
        pricing_type="fixed",
        cash_index=-1
    )

    search_space = {
        'delta': {'type': 'float', 'low': 0.05, 'high': 0.99},
        'cost_penalty': {'type': 'float', 'low': 0.0, 'high': 1.0},
        'alpha': {'type': 'float', 'low': 0.0, 'high': 0.15},
    }

    storage = make_optuna_storage(db_path="ons_hpo.db")

    experiment = ONSOptunaExperiment(
        cost_model=cost_model,
        model_name=f"IB Fixed (${price_to_k}k)[AAPL MSFT GOOGL AMZN META]", # NVDA TSLA AMD PLTR SNOW
        stocks=data_dict['stock_names'],
        train_data=data_dict['train_price_relatives'],
        val_data=data_dict['val_price_relatives'],
        test_data=data_dict['test_price_relatives'],
        train_prices=data_dict['train_actual_prices'],
        val_prices=data_dict['val_actual_prices'],
        test_prices=data_dict['test_actual_prices'],
        initial_capital=INITIAL_CAPITAL,
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

    # BAH benchmark on test period using ONS initial portfolio
    bah_ons = run_buy_and_hold(
        data_dict['test_price_relatives'],
        initial_weights=experiment.test_results['ons_initial_portfolio'],
    )
    bah_uniform = run_buy_and_hold(data_dict['test_price_relatives'])

    print(f"Uniform BAH : {bah_uniform['final_wealth']:.4f}")
    print(f"BAH (ONS init): {bah_ons['final_wealth']:.4f}")
    print(f"ONS: {experiment.test_results['final_wealth']:.4f}")

    if SAVE_RESULTS:
        results_dir = Path(RESULTS_BASE_DIR) / f"optuna_{len(STOCKS)}_{datetime.now().strftime('%d-%m-%y_%H-%M-%S')}"

        save_experiment_results(
            results_dir=results_dir,
            experiments={experiment.model_name: experiment},
            data_dict=data_dict,
            run_info={'hpo_method': 'optuna', 'n_trials': N_TRIALS, 'n_jobs': N_JOBS, 'db_path': 'ons_hpo.db',},
            bah_results={'uniform': bah_uniform, 'ons_initial': bah_ons,},
        )