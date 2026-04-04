import copy

from utils.data_prep import prepare_stock_data_3split
from benchmarks.portfolio.buy_and_hold import run_buy_and_hold


def run_stability_analysis(experiment_class,cost_model, stocks, initial_capital, optimize_metric, search_space, windows,
    n_trials=40, n_jobs=10, verbose=False,):
    """
    Run HPO + test across multiple time windows to check hyperparameter
    and performance stability.

    experiment_class : the subclass to instantiate (e.g. ONSOptunaExperiment)
    windows          : list of (train_start, train_end, val_end, test_end) tuples
    """
    stability_results = []

    for i, (train_start, train_end, val_end, test_end) in enumerate(windows, 1):
        print(f"\n{'='*60}")
        print(f"WINDOW {i}: {train_start} → train_end={train_end} | val_end={val_end} | test_end={test_end}")
        print(f"{'='*60}")

        data_dict = prepare_stock_data_3split(
            stocks=stocks,
            train_start_date=train_start,
            train_end_date=train_end,
            val_end_date=val_end,
            test_end_date=test_end,
            include_index_benchmarks=False,
            include_cash=(stocks[-1].lower() == 'cash'),
        )

        exp = experiment_class(
            cost_model=copy.deepcopy(cost_model),
            model_name=f"stability_w{i}",
            stocks=data_dict['stock_names'],
            train_data=data_dict['train_price_relatives'],
            val_data=data_dict['val_price_relatives'],
            test_data=data_dict['test_price_relatives'],
            train_prices=data_dict['train_actual_prices'],
            val_prices=data_dict['val_actual_prices'],
            test_prices=data_dict['test_actual_prices'],
            initial_capital=initial_capital,
            optimize_metric=optimize_metric,
        )

        best_params = exp.optuna_search(
            search_space=search_space,
            n_trials=n_trials,
            n_jobs=n_jobs,
            storage=None,
            verbose=verbose,
        )

        test_results = exp.run_test(best_params, verbose=False)
        bah_uniform = run_buy_and_hold(data_dict['test_price_relatives'])

        stability_results.append({
            'window': i,
            'train_start': train_start,
            'train_end': train_end,
            'val_end': val_end,
            'test_end': test_end,
            'best_params': best_params,
            'final_wealth': test_results['final_wealth'],
            'sharpe': test_results['sharpe'],
            'sortino': test_results['sortino'],
            'max_drawdown': test_results['max_drawdown'],
            'trade_freq': test_results['trade_frequency'],
            'cost_drag': test_results['cost_drag_pct'],
            'val_dsr': test_results['val_dsr'],
            'test_dsr': test_results['test_dsr'],
            'n_trials': test_results['n_trials'],
            'bah_final_wealth': bah_uniform['final_wealth'],
        })

    _print_stability_summary(stability_results)
    return stability_results


def _print_stability_summary(stability_results):
    print(f"\n{'='*72}")
    print("STABILITY SUMMARY")
    print(f"{'='*72}")

    header = (f"{'Win':<4} {'Train Start':>11} {'Test End':>10} "
              f"{'Wealth':>7} {'BAH':>7} {'Sharpe':>7} "
              f"{'Sortino':>7} {'MDD':>6} {'TrdFq':>6} {'ValDSR':>7}")
    print(header)
    print('-' * len(header))

    for r in stability_results:
        print(
            f"{r['window']:<4} {r['train_start']:>11} {r['test_end']:>10} "
            f"{r['final_wealth']:>7.4f} {r['bah_final_wealth']:>7.4f} "
            f"{r['sharpe']:>7.4f} {r['sortino']:>7.4f} "
            f"{r['max_drawdown']:>6.4f} {r['trade_freq']:>6.2%} "
            f"{r['val_dsr']:>7.4f}"
        )

    if not stability_results:
        return

    param_keys = list(stability_results[0]['best_params'].keys())
    print(f"\nHYPERPARAMETER STABILITY:")
    col_w = 10
    header2 = f"{'Win':<4}" + "".join(f"{k:>{col_w}}" for k in param_keys)
    print(header2)
    print('-' * len(header2))
    for r in stability_results:
        row = f"{r['window']:<4}"
        for k in param_keys:
            row += f"{r['best_params'][k]:>{col_w}.4f}"
        print(row)