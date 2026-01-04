from datetime import datetime, timedelta
import json

import yfinance as yf
import numpy as np

def prepare_stock_data(stocks, train_days=730, test_days=365, end_date=None):
    if end_date is None:
        end_date = datetime.now()

    # Calculate start dates
    test_start = end_date - timedelta(days=test_days)
    train_start = test_start - timedelta(days=train_days)

    print(f"Downloading data for {len(stocks)} stocks")
    print(f"Train period: {train_start.date()} to {test_start.date()} ({train_days} calendar days)")
    print(f"Test period: {test_start.date()} to {end_date.date()} ({test_days} calendar days)")

    # Download data
    data = yf.download(stocks, start=train_start, end=end_date, auto_adjust=True)['Close']

    if len(stocks) == 1:
        data = data.to_frame(name=stocks[0])

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

    return {
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