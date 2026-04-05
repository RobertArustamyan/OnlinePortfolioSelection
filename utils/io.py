import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


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
            'regime_analysis': experiment.regime_results if hasattr(experiment, 'regime_results') else None,
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