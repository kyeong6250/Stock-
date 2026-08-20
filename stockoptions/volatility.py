"""Volatility statistics: realized vol from a price history, and where a
current implied vol / strike sits relative to its own history or chain.

These are pure functions over pandas/numpy data -- no network calls -- so
they're testable with synthetic data and reusable regardless of where the
prices or IVs actually came from (yfinance today, something else later).
"""

import numpy as np
import pandas as pd


def historical_volatility(prices: pd.Series, window: int | None = None, trading_days: int = 252) -> float:
    """Annualized realized volatility from a price series (stdev of daily
    log returns, annualized by sqrt(trading_days)). If `window` is given,
    only the most recent `window` log returns are used."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if window is not None:
        log_returns = log_returns.tail(window)
    if len(log_returns) < 2:
        raise ValueError("need at least 2 log returns to compute volatility")
    return float(log_returns.std(ddof=1) * np.sqrt(trading_days))


def iv_rank(current_iv: float, iv_history) -> float:
    """0-100: where current_iv sits between the historical min and max.
    100 = highest IV seen in the lookback window, 0 = lowest."""
    arr = np.asarray(iv_history, dtype=float)
    if len(arr) == 0:
        raise ValueError("iv_history must not be empty")
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return 50.0  # no historical range to rank against -- call it neutral
    return float(100 * (current_iv - lo) / (hi - lo))


def iv_percentile(current_iv: float, iv_history) -> float:
    """0-100: the percentage of historical observations below current_iv.
    Unlike iv_rank, this isn't distorted by a single extreme outlier day."""
    arr = np.asarray(iv_history, dtype=float)
    if len(arr) == 0:
        raise ValueError("iv_history must not be empty")
    return float(100 * np.mean(arr < current_iv))


def iv_skew_zscore(strike_ivs) -> pd.Series:
    """Z-score of each strike's IV against the mean/stdev of the full chain.
    Flags strikes trading unusually rich or cheap relative to the smile --
    an educational mispricing signal, not a real, actionable arbitrage
    (any genuine gap gets closed by market makers before a retail script
    would ever see it)."""
    s = strike_ivs if isinstance(strike_ivs, pd.Series) else pd.Series(strike_ivs)
    if len(s) == 0:
        raise ValueError("strike_ivs must not be empty")
    std = s.std(ddof=1)
    if not std or np.isnan(std):
        return s * 0.0  # all IVs identical (or only one strike): no skew to speak of
    return (s - s.mean()) / std
