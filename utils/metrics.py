import numpy as np
from scipy.special import ndtr


def compute_risk_metrics(daily_wealth):
    """
    Compute risk-adjusted performance metrics from a daily wealth series.
    Returns a dict with sharpe, sortino, max_drawdown, calmar, annualized_return.
    """
    daily_wealth = np.array(daily_wealth)
    daily_returns = np.diff(daily_wealth) / daily_wealth[:-1]
    n_days = len(daily_returns)

    # Sharpe Ratio (annualized)
    if n_days > 1 and np.std(daily_returns) > 1e-10:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino Ratio (annualized, using downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1:
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        sortino = (np.mean(daily_returns) / downside_std) * np.sqrt(252) if downside_std > 1e-10 else 0.0
    else:
        sortino = 0.0

    # Max Drawdown
    cumulative = daily_wealth / daily_wealth[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (running_max - cumulative) / running_max
    max_drawdown = float(np.max(drawdowns))

    # Annualized Return
    annualized_return = (daily_wealth[-1] / daily_wealth[0]) ** (252 / n_days) - 1 if n_days > 0 else 0.0

    # Calmar Ratio (annualized return / max drawdown)
    calmar = annualized_return / max_drawdown if max_drawdown > 1e-10 else 0.0

    return {
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_drawdown': round(float(max_drawdown), 4),
        'calmar': round(float(calmar), 4),
        'annualized_return': round(float(annualized_return), 4),
    }


def compute_dsr(sharpe_obs, n_trials, n_days, skewness, excess_kurtosis):
    """
    Deflated Sharpe Ratio.
    Corrects for non-normality and multiple testing.
    """
    # Expected max Sharpe under null across n_trials
    e_max_sr = ((1 - np.euler_gamma) * ndtr(1 - 1 / n_trials) + np.euler_gamma * ndtr(1 - 1 / (n_trials * np.e)))

    # Variance of SR estimator adjusted for non-normality
    sr_std = np.sqrt((1 - skewness * sharpe_obs + (excess_kurtosis - 1) / 4 * sharpe_obs ** 2) / (n_days - 1))

    if sr_std < 1e-10:
        return 0.0

    dsr = ndtr((sharpe_obs - e_max_sr) / sr_std)
    return round(float(dsr), 4)