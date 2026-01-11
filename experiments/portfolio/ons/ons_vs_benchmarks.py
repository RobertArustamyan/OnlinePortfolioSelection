import numpy as np

from algorithms.portfolio.online_newton_step import OnlineNewtonStep
from utils.compare_strategies import compare_strategies
from utils.data_prep import prepare_stock_data

if __name__ == '__main__':
    # Settings for saving results
    save_plot = False
    path = '../../plots/portfolio'
    fig_name = 'ons_5stocks'

    # Flag to control wealth continuation
    continue_wealth_from_training = False

    data_dict = prepare_stock_data(
        stocks=["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"],
        train_days=700,
        test_days=350
    )

    train_price_relatives = data_dict['train_price_relatives']
    test_price_relatives = data_dict['test_price_relatives']
    stocks = data_dict['stock_names']

    T = len(train_price_relatives) + len(test_price_relatives) if continue_wealth_from_training else len(test_price_relatives)
    n = len(stocks)
    eta = (n ** 1.25) / (np.sqrt(T * np.log(n * T)))
    beta = 1.0 / (8 * (n ** 0.25) * np.sqrt(T * np.log(n * T)))

    portfolio = OnlineNewtonStep(
        n_stocks=n,
        eta=eta,
        beta=beta,
        delta=0.8
    )

    print("\nTraining\n")

    train_results = portfolio.simulate_trading(train_price_relatives, verbose=True)
    train_wealth = train_results['final_wealth']

    if continue_wealth_from_training:
        test_daily_wealth = [train_results['daily_wealth'][-1]]
        test_wealth_current = train_results['daily_wealth'][-1]
        print(f"Continuing from training wealth: {test_wealth_current:.4f}")
    else:
        test_daily_wealth = [1.0]
        test_wealth_current = 1.0
        print("Starting fresh with wealth = 1.0")

    for day, price_rel in enumerate(test_price_relatives):
        current_portfolio = portfolio.get_portfolio()
        daily_return = np.dot(current_portfolio, price_rel)
        test_wealth_current *= daily_return
        test_daily_wealth.append(test_wealth_current)
        portfolio.update(price_rel, current_portfolio)

        if (day + 1) % 10 == 0 or day == 0:
            print(f"Test Day {day + 1}: Wealth = {test_wealth_current:.4f}, " +
                  f"Portfolio = [{', '.join([f'{x:.3f}' for x in current_portfolio])}]")

        # Prepare results for comparison
    if continue_wealth_from_training:
        # Combine train and test wealth
        combined_daily_wealth = np.concatenate([
            train_results['daily_wealth'][:-1],
            test_daily_wealth
        ])
        ons_results = {
            'strategy': 'ONS',
            'daily_wealth': combined_daily_wealth,
            'final_wealth': test_wealth_current
        }
        price_relatives_for_plot = np.vstack([train_price_relatives, test_price_relatives])
    else:
        # Test only
        ons_results = {
            'strategy': 'ONS (trained on 2yr)',
            'daily_wealth': np.array(test_daily_wealth),
            'final_wealth': test_wealth_current
        }
        price_relatives_for_plot = test_price_relatives

        # Compare strategies
    results = compare_strategies(
        price_relatives_for_plot,
        stock_names=stocks,
        show_plot=True,
        save_path=f'{path}/{fig_name}.png' if save_plot else None,
        crp_weights=np.ones(len(stocks)) / len(stocks),
        additional_results=ons_results
    )