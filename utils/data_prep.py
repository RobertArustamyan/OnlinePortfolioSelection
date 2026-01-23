from datetime import datetime, timedelta
import json

import yfinance as yf
import numpy as np
import pandas as pd


def prepare_stock_data(stocks, train_start_date, train_end_date, test_end_date=None, include_benchmarks=True):
    """
    Prepare stock data for algorithms.

    Args:
        stocks: List of stock tickers
        train_start_date: Start date for training period (str or datetime)
        train_end_date: End date for training period = Start date for test period (str or datetime)
        test_end_date: End date for test period (str or datetime). If None, uses today's date
        include_benchmarks: Whether to download benchmark data
    """

    # Convert dates
    train_start = pd.to_datetime(train_start_date)
    test_start = pd.to_datetime(train_end_date)  # train_end = test_start

    if test_end_date is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(test_end_date)

    train_days = (test_start - train_start).days
    test_days = (end_date - test_start).days

    print(f"Downloading data for {len(stocks)} stocks")
    print(f"Train period: {train_start.date()} to {test_start.date()} ({train_days} calendar days)")
    print(f"Test period: {test_start.date()} to {end_date.date()} ({test_days} calendar days)")

    # Download data
    data = yf.download(stocks, start=train_start, end=end_date, auto_adjust=True)

    if len(stocks) == 1:
        data = data[['Close']].rename(columns={'Close': stocks[0]})
    else:
        data = data['Close']

    # Calculate price relatives
    price_relatives_df = (data / data.shift(1)).dropna()

    # Get actual prices (align with price relatives)
    actual_prices_df = data.loc[price_relatives_df.index]

    # Split train and test
    train_mask = price_relatives_df.index < test_start
    test_mask = ~train_mask

    # Price relatives
    train_price_relatives_df = price_relatives_df[train_mask]
    test_price_relatives_df = price_relatives_df[test_mask]

    # Actual prices
    train_actual_prices_df = actual_prices_df[train_mask]
    test_actual_prices_df = actual_prices_df[test_mask]

    # Convert to numpy arrays
    train_price_relatives = train_price_relatives_df.values
    test_price_relatives = test_price_relatives_df.values
    train_actual_prices = train_actual_prices_df.values
    test_actual_prices = test_actual_prices_df.values

    print(f"\nTrading days:")
    print(f"Train: {len(train_price_relatives)} days")
    print(f"Test: {len(test_price_relatives)} days")

    result = {
        'train_price_relatives': train_price_relatives,
        'test_price_relatives': test_price_relatives,
        'train_actual_prices': train_actual_prices,
        'test_actual_prices': test_actual_prices,
        'train_dates': train_price_relatives_df.index,
        'test_dates': test_price_relatives_df.index,
        'train_price_dates': train_actual_prices_df.index,
        'test_price_dates': test_actual_prices_df.index,
        'stock_names': stocks,
        'data': data,
        'split_date': test_start
    }

    if include_benchmarks:
        print(f"Downloading benchmark data")
        benchmark_tickers = {
            'NASDAQ': '^IXIC',
            'SP500': '^GSPC'
        }

        benchmark_data = {}

        for name, ticker in benchmark_tickers.items():
            try:
                bench_data = yf.download(ticker, start=train_start, end=end_date, auto_adjust=True)['Close']
                bench_relatives_df = (bench_data / bench_data.shift(1)).dropna()

                bench_test_relatives_df = bench_relatives_df[
                    bench_relatives_df.index.isin(test_price_relatives_df.index)]

                benchmark_data[name] = bench_test_relatives_df.values.reshape(-1, 1)

            except Exception as e:
                print(f"Could not download {name} ({ticker}): {e}")
                benchmark_data[name] = None

        result['benchmark_test_relatives'] = benchmark_data

    return result


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and types"""

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