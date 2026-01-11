import numpy as np
import matplotlib.pyplot as plt

from data_generators.simple_data_generator import generate_stock_data
from algorithms.threshold.local_extrema_volatility import LocalExtremaVolatilityTrader
from algorithms.threshold.adaptive_threshold import AdaptiveThresholdTrader
from benchmarks.threshold.oracle_trader import hindsight_optimal_wealth_daily_updated, hindsight_optimal_wealth_onsale_updated


if __name__ == "__main__":
    # Settings for saving results
    save_plot = True
    path = '../../plots/threshold'
    fig_name = 'adaptive_extrema'

    np.random.seed(42)

    total_days = 500

    # Generate oscillating  stock
    oscillating_stock = generate_stock_data(
        initial_value=100,
        total_steps=total_days,
        min_oscillation_steps=1,
        max_oscillation_steps=6,
        min_flat_steps=3,
        max_flat_steps=10,
        step_size=0.2
    )
    current_day_index = total_days - 150

    historical_data = oscillating_stock[:current_day_index] # Training data
    future_data = oscillating_stock[current_day_index:] # Testing data

    wealth1, daily_wealth1 = hindsight_optimal_wealth_onsale_updated(historical_data)
    wealth2, daily_wealth2 = hindsight_optimal_wealth_daily_updated(historical_data)


    adaptive_trader1 = AdaptiveThresholdTrader(lookback_window=60, percentile_buy=25, percentile_sell=75, adaptation_rate=0.1)
    adaptive_trader2 = AdaptiveThresholdTrader(lookback_window=60, percentile_buy=20, percentile_sell=80, adaptation_rate=0.1)
    adaptive_trader3 = AdaptiveThresholdTrader(lookback_window=60, percentile_buy=10, percentile_sell=90, adaptation_rate=0.1)

    extrema_trader1 = LocalExtremaVolatilityTrader(lookback_window=20, volatility_window=10, buy_zscore=-1.0, sell_zscore=1.0)
    extrema_trader2 = LocalExtremaVolatilityTrader(lookback_window=40, volatility_window=15, buy_zscore=-1.5, sell_zscore=1.5)
    extrema_trader3 = LocalExtremaVolatilityTrader(lookback_window=30, volatility_window=10, buy_zscore=-0.8, sell_zscore=0.8)
    extrema_trader4 = LocalExtremaVolatilityTrader(lookback_window=60, volatility_window=5, buy_zscore=-0.8, sell_zscore=0.8)


    all_traders = [adaptive_trader1, adaptive_trader2, adaptive_trader1, extrema_trader1, extrema_trader2, extrema_trader3, extrema_trader4]
    trader_names = ["Adaptive1: LB=60, P=25/75","Adaptive2: LB=60, P=20/70", "Adaptive3: LB=60, P=10/90",
        "Extrema1: LB=20, V=10, Z=-+1.0", "Extrema2: LB=40, V=15, Z=-+1.5", "Extrema3: LB=30, V=10, Z=-+0.8",
                    "Extrema4: LB=60, Z=-+0.8,"
    ]

    results = []
    for i, (trader, name) in enumerate(zip(all_traders, trader_names)):
        print(f"\n{name}")
        trader.learn_initial_thresholds(historical_data)
        final_wealth, daily_wealth, decisions = trader.trade_online(future_data)
        results.append((name, final_wealth, daily_wealth, decisions))


    fig, (ax_all, ax_price) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    _, daily_wealth_hindsight_daily = hindsight_optimal_wealth_daily_updated(future_data)

    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

    for (name, wealth, daily_wealth, decisions), color in zip(results, colors):
        ax_all.plot(
            daily_wealth,
            label=f"{name} (${wealth:.2f})",
            linewidth=1.6,
            color=color,
            alpha=0.9
        )

    ax_all.plot(
        daily_wealth_hindsight_daily,
        color="orange",
        linestyle="--",
        linewidth=2.5,
        label="Hindsight"
    )

    ax_all.set_ylabel("Wealth")
    ax_all.set_title("Comparison of Different Approaches")
    ax_all.legend(fontsize=7, ncol=2)
    ax_all.grid(True, alpha=0.3)

    ax_price.plot(future_data, color="black", linewidth=2)
    ax_price.set_title("Future Stock Price")
    ax_price.set_xlabel("Day")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_plot:
        plt.savefig(f"{path}/{fig_name}.png")
    plt.show()