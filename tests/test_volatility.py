import numpy as np
import pandas as pd
import pytest

from stockoptions.volatility import (
    historical_volatility,
    iv_percentile,
    iv_rank,
    iv_skew_zscore,
)


def test_historical_volatility_matches_numpy_stdev_of_log_returns():
    prices = pd.Series([100, 102, 101, 105, 103, 107, 110, 108])
    log_returns = np.log(prices / prices.shift(1)).dropna()
    expected = log_returns.std(ddof=1) * np.sqrt(252)
    assert historical_volatility(prices) == pytest.approx(expected)


def test_historical_volatility_scales_linearly_with_return_magnitude():
    # Doubling every daily log-return's magnitude should double the stdev,
    # and therefore double the annualized volatility.
    base_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02, -0.01])
    prices_a = pd.Series(np.exp(np.concatenate([[0], np.cumsum(base_returns)])))
    prices_b = pd.Series(np.exp(np.concatenate([[0], np.cumsum(base_returns * 2)])))
    vol_a = historical_volatility(prices_a)
    vol_b = historical_volatility(prices_b)
    assert vol_b == pytest.approx(vol_a * 2, rel=1e-6)


def test_historical_volatility_respects_window():
    # A long flat run followed by a volatile tail: windowed vol should be
    # much higher than the full-series vol, since it only sees the tail.
    flat = [100.0] * 50
    volatile = [100, 110, 95, 115, 90, 120, 85, 125]
    prices = pd.Series(flat + volatile)
    full_vol = historical_volatility(prices)
    windowed_vol = historical_volatility(prices, window=7)
    assert windowed_vol > full_vol


def test_historical_volatility_requires_at_least_two_returns():
    with pytest.raises(ValueError):
        historical_volatility(pd.Series([100.0]))


@pytest.mark.parametrize("current,expected", [(10, 0.0), (20, 100.0), (15, 50.0)])
def test_iv_rank_at_min_max_and_midpoint(current, expected):
    history = [10, 12, 14, 16, 18, 20]
    assert iv_rank(current, history) == pytest.approx(expected)


def test_iv_rank_is_neutral_when_history_has_no_range():
    assert iv_rank(0.25, [0.30, 0.30, 0.30]) == 50.0


def test_iv_percentile_counts_observations_strictly_below():
    history = [10, 20, 30, 40, 50]
    assert iv_percentile(35, history) == pytest.approx(60.0)  # 3 of 5 below 35
    assert iv_percentile(5, history) == pytest.approx(0.0)
    assert iv_percentile(55, history) == pytest.approx(100.0)


def test_iv_skew_zscore_uniform_ivs_gives_zero_skew():
    result = iv_skew_zscore({90: 0.30, 100: 0.30, 110: 0.30})
    assert (result == 0).all()


def test_iv_skew_zscore_flags_the_richest_strike_highest():
    result = iv_skew_zscore({90: 0.28, 100: 0.30, 110: 0.45})
    assert result[110] > result[100] > result[90]


def test_iv_skew_zscore_rejects_empty_input():
    with pytest.raises(ValueError):
        iv_skew_zscore({})
