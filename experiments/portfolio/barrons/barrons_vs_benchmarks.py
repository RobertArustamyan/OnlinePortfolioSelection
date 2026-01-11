import numpy as np

from algorithms.portfolio.barrons import Barrons
from utils.compare_strategies import compare_strategies
from utils.data_prep import prepare_stock_data

if __name__ == '__main__':
    save_plot = False
    path = '../../plots/portfolio'
    fig_name = 'barrons_4stocks'

    continue_wealth_from_training = False

    data_dict = prepare_stock_data(
        stocks=["AAPL", "TSLA", "MSFT", "NVDA"],
        train_days=7000,
        test_days=360
    )

    train_price_relatives = data_dict['train_price_relatives']
    test_price_relatives = data_dict['test_price_relatives']
    stocks = data_dict['stock_names']

    T = len(train_price_relatives) + len(test_price_relatives) if continue_wealth_from_training else len(test_price_relatives)
    n = len(stocks)

    portfolio = Barrons(
        n_stocks=n,
        beta=0.4,
        eta=0.1,
        T=T
    )

    print("\nTraining\n")
    train_results = portfolio.simulate_trading(train_price_relatives, verbose=True)
    train_wealth = train_results['final_wealth']

    # Initialize test wealth
    if continue_wealth_from_training:
        test_daily_wealth = [train_results['daily_wealth'][-1]]
        test_wealth_current = train_results['daily_wealth'][-1]
        print(f"Continuing from training wealth: {test_wealth_current:.4f}")
    else:
        test_daily_wealth = [1.0]
        test_wealth_current = 1.0
        print("Starting fresh with wealth = 1.0")

    print("\nTesting\n")
    for day, price_rel in enumerate(test_price_relatives):
        current_portfolio = portfolio.get_portfolio()
        daily_return = np.dot(current_portfolio, price_rel)
        test_wealth_current *= daily_return
        test_daily_wealth.append(test_wealth_current)

        # Update portfolio with last used portfolio
        portfolio.update(price_rel)

        if (day + 1) % 10 == 0 or day == 0:
            print(f"Test Day {day + 1}: Wealth = {test_wealth_current:.4f}, " +
                  f"Portfolio = [{', '.join([f'{x:.3f}' for x in current_portfolio])}]")

    # Prepare results for comparison
    if continue_wealth_from_training:
        combined_daily_wealth = np.concatenate([
            train_results['daily_wealth'][:-1],
            test_daily_wealth
        ])
        barrons_results = {
            'strategy': 'Barrons',
            'daily_wealth': combined_daily_wealth,
            'final_wealth': test_wealth_current
        }
        price_relatives_for_plot = np.vstack([train_price_relatives, test_price_relatives])
    else:
        barrons_results = {
            'strategy': 'Barrons (trained on 2yr)',
            'daily_wealth': np.array(test_daily_wealth),
            'final_wealth': test_wealth_current
        }
        price_relatives_for_plot = test_price_relatives

    results = compare_strategies(
        price_relatives_for_plot,
        stock_names=stocks,
        show_plot=True,
        save_path=f'{path}/{fig_name}.png' if save_plot else None,
        crp_weights=np.ones(len(stocks)) / len(stocks),
        additional_results=barrons_results
    )
