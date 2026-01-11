import numpy as np

from algorithms.portfolio.barrons import Barrons
from algorithms.portfolio.online_newton_step import OnlineNewtonStep
from algorithms.portfolio.ada_barrons import AdaBarrons
from utils.compare_strategies import compare_strategies
from utils.data_prep import prepare_stock_data

if __name__ == '__main__':
    save_plot = False
    path = '../../plots/portfolio'
    fig_name = 'barrons_vs_ons_vs_ada_barrons'

    continue_wealth_from_training = False

    data_dict = prepare_stock_data(
        stocks=["AAPL", "NVDA", "MSFT", "AMZN"],
        train_days=5100,
        test_days=2000
    )

    train_price_relatives = data_dict['train_price_relatives']
    test_price_relatives = data_dict['test_price_relatives']
    stocks = data_dict['stock_names']

    T = len(train_price_relatives) + len(test_price_relatives) if continue_wealth_from_training else len(
        test_price_relatives)
    n = len(stocks)

    # Initialize portfolios
    barrons_portfolio = Barrons(
        n_stocks=n,
        beta=0.4,
        eta=0.1,
        T=T
    )

    ada_barrons_portfolio = AdaBarrons(
        n_stocks=n,
        T=T
    )

    eta_ons = (n ** 1.25) / (np.sqrt(T * np.log(n * T)))
    beta_ons = 1.0 / (8 * (n ** 0.25) * np.sqrt(T * np.log(n * T)))

    ons_portfolio = OnlineNewtonStep(
        n_stocks=n,
        eta=eta_ons,
        beta=beta_ons,
        delta=0.8
    )

    print("\nTraining Barrons\n")
    barrons_train_results = barrons_portfolio.simulate_trading(train_price_relatives, verbose=True)

    print("\nTraining Ada-Barrons\n")
    ada_barrons_train_results = ada_barrons_portfolio.simulate_trading(train_price_relatives, verbose=True)
    print(f"Ada-Barrons training: {ada_barrons_train_results['restart_count']} restarts, "
          f"final beta={ada_barrons_train_results['final_beta']:.6f}")

    print("\nTraining ONS\n")
    ons_train_results = ons_portfolio.simulate_trading(train_price_relatives, verbose=True)

    if continue_wealth_from_training:
        barrons_test_wealth = [barrons_train_results['daily_wealth'][-1]]
        ada_barrons_test_wealth = [ada_barrons_train_results['daily_wealth'][-1]]
        ons_test_wealth = [ons_train_results['daily_wealth'][-1]]
    else:
        barrons_test_wealth = [1.0]
        ada_barrons_test_wealth = [1.0]
        ons_test_wealth = [1.0]

    print("\nTesting Barrons, Ada-Barrons and ONS\n")
    for day, price_rel in enumerate(test_price_relatives):
        # Barrons
        b_portfolio = barrons_portfolio.get_portfolio()
        barrons_test_wealth_current = barrons_test_wealth[-1] * np.dot(b_portfolio, price_rel)
        barrons_test_wealth.append(barrons_test_wealth_current)
        barrons_portfolio.update(price_rel)

        # Ada-Barrons
        ab_portfolio = ada_barrons_portfolio.get_portfolio()
        ada_barrons_test_wealth_current = ada_barrons_test_wealth[-1] * np.dot(ab_portfolio, price_rel)
        ada_barrons_test_wealth.append(ada_barrons_test_wealth_current)
        ada_barrons_portfolio.update(price_rel, ab_portfolio)

        # ONS
        o_portfolio = ons_portfolio.get_portfolio()
        ons_test_wealth_current = ons_test_wealth[-1] * np.dot(o_portfolio, price_rel)
        ons_test_wealth.append(ons_test_wealth_current)
        ons_portfolio.update(price_rel, o_portfolio)

        if (day + 1) % 10 == 0 or day == 0:
            print(f"Day {day + 1}: Barrons={barrons_test_wealth_current:.4f}, "
                  f"Ada-Barrons={ada_barrons_test_wealth_current:.4f} (beta={ada_barrons_portfolio.beta:.4f}), "
                  f"ONS={ons_test_wealth_current:.4f}")

    print(f"\nAda-Barrons final stats: {ada_barrons_portfolio.restart_count} total restarts, "
          f"final beta={ada_barrons_portfolio.beta:.6f}")

    if continue_wealth_from_training:
        barrons_combined_wealth = np.concatenate([
            barrons_train_results['daily_wealth'][:-1],
            barrons_test_wealth
        ])

        ada_barrons_combined_wealth = np.concatenate([
            ada_barrons_train_results['daily_wealth'][:-1],
            ada_barrons_test_wealth
        ])

        ons_combined_wealth = np.concatenate([
            ons_train_results['daily_wealth'][:-1],
            ons_test_wealth
        ])

        barrons_results = {
            'strategy': 'Barrons (β=0.4, η=0.1)',
            'daily_wealth': barrons_combined_wealth,
            'final_wealth': barrons_test_wealth[-1]
        }

        ada_barrons_results = {
            'strategy': f'Ada-Barrons (adaptive β, {ada_barrons_portfolio.restart_count} restarts)',
            'daily_wealth': ada_barrons_combined_wealth,
            'final_wealth': ada_barrons_test_wealth[-1]
        }

        ons_results = {
            'strategy': 'ONS',
            'daily_wealth': ons_combined_wealth,
            'final_wealth': ons_test_wealth[-1]
        }

        combined_price_relatives = np.vstack([train_price_relatives, test_price_relatives])

        results = compare_strategies(
            price_relatives=combined_price_relatives,
            stock_names=stocks,
            show_plot=True,
            save_path=f'{path}/{fig_name}.png' if save_plot else None,
            crp_weights=np.ones(len(stocks)) / len(stocks),
            additional_results=[barrons_results, ada_barrons_results, ons_results]
        )
    else:
        barrons_results = {
            'strategy': 'Barrons (β=0.4, η=0.1)',
            'daily_wealth': np.array(barrons_test_wealth),
            'final_wealth': barrons_test_wealth[-1]
        }

        ada_barrons_results = {
            'strategy': f'Ada-Barrons (adaptive β, {ada_barrons_portfolio.restart_count} restarts)',
            'daily_wealth': np.array(ada_barrons_test_wealth),
            'final_wealth': ada_barrons_test_wealth[-1]
        }

        ons_results = {
            'strategy': 'ONS',
            'daily_wealth': np.array(ons_test_wealth),
            'final_wealth': ons_test_wealth[-1]
        }

        results = compare_strategies(
            price_relatives=test_price_relatives,
            stock_names=stocks,
            show_plot=True,
            save_path=f'{path}/{fig_name}.png' if save_plot else None,
            crp_weights=np.ones(len(stocks)) / len(stocks),
            additional_results=[barrons_results, ada_barrons_results, ons_results]
        )
