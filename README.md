# CURRENT VERSION OF README IS OUTDATED.

# Online Portfolio Selection

## Project Structure

```
.
├── algorithms/
│   ├── portfolio/
│   │   ├── r_universal.py                    # R-Universal Portfolio
│   │   ├── online_newton_step.py             # Online Newton Step (ONS)
│   │   ├── online_newton_step_costs.py       # Cost-aware ONS
│   │   ├── barrons.py                        # BARRONS
│   │   ├── barrons_costs.py                  # Cost-aware BARRONS
│   │   ├── ada_barrons.py                    # Ada-BARRONS
│   │   └── ada_barrons_costs.py              # Cost-aware Ada-BARRONS
│   └── transactions/
│       ├── cost.py                           # Base transaction cost interface
│       ├── interactive_brokers.py            # Interactive Brokers commission model
│       ├── fixed_cost_per_asset.py           # Fixed cost per asset
│       ├── fixed_cost_per_rebalancing.py     # Fixed cost per rebalancing
│       ├── linear_cost.py                    # Linear transaction costs
│       └── quadratic_cost.py                 # Quadratic transaction costs
│
├── benchmarks/
│   └── portfolio/
│       ├── buy_and_hold.py                   # Buy and hold strategy
│       ├── constant_rebalanced_portfolio.py  # Constant rebalanced portfolio
│       └── best_constant_rebalanced_portfolio.py  # Best CRP in hindsight
│
├── experiments/
│   └── portfolio/
│       ├── ons/
│       │   └── ons_cost_vs_benchmarks.py     # ONS experiments
│       └── barrons/
│           └── barrons_cost_vs_benchmarks.py # BARRONS experiments
│
├── results/
│   └── portfolio/
│       └── WithCosts/
│           ├── ONS/                          # ONS results
│           ├── BARRONS/                      # BARRONS results
│           └── AdaBARRONS/                   # Ada-BARRONS results
│
└── utils/
    └── data_prep.py                          # Stock data preparation
```

## Usage

Run experiments from the `experiments/` directory:

```bash
# ONS with transaction costs
python experiments/portfolio/ons/ons_cost_vs_benchmarks.py

# BARRONS with transaction costs
python experiments/portfolio/barrons/barrons_cost_vs_benchmarks.py
```

Results are saved to `results/portfolio/WithCosts/` with:
- Experiment metadata (JSON)
- All hyperparameter combinations (CSV)
- Detailed results including daily wealth, portfolios, and costs (JSON)

## Dependencies

**Python:** 3.12.3

Install required packages:

```bash
pip install -r requirements.txt
```

## References

- Cover, T. M. (1991). Universal Portfolios. *Mathematical Finance*
- Agarwal, A., Hazan, E., Kale, S., & Schapire, R. E. (2006). Algorithms for portfolio management based on the Newton method
- Luo, H., Wei, C.-Y., & Zheng, K. (2018). Efficient online portfolio with logarithmic regret
- Kalai, A., & Vempala, S. (2002). Efficient algorithms for universal portfolios
