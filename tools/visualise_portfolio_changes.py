import json

import numpy as np

from utils.results_plotter import ResultsPlotter

with open(r"C:\Users\rarustamyan\PycharmProjects\UniversalPortfolios\results\portfolio\WithCosts\BARRONS\TrainingTimeAnalysis\train3600_test730_stocks4_13-01-26_05-18-52\experiment_metadata.json", "r") as f:
    data = f.read()

data = json.loads(data)
ticker_names = data['experiment_settings']['stocks']

with open(r"C:\Users\rarustamyan\PycharmProjects\UniversalPortfolios\results\portfolio\WithCosts\BARRONS\TrainingTimeAnalysis\train3600_test730_stocks4_13-01-26_05-18-52\barrons_costs_detailed_results.json", "r") as f:
    data = f.read()

data = json.loads(data)

detailed_results = data['experiments']['IB Fixed ($1.0k)'][0]
plotter = ResultsPlotter(results=None, model_name=None, detailed_results=detailed_results)

n_days = len(detailed_results['testing']['portfolios_used'])
sample_days = np.linspace(0, n_days - 1, 5, dtype=int)

plotter.plot_portfolio_evolution(ticker_names=ticker_names, sample_days=sample_days)