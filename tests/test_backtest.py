import numpy as np
import pandas as pd
import pytest

from stockoptions.backtest import _max_drawdown, _sharpe, backtest


def test_max_drawdown_matches_hand_computed_value():
    equity = pd.Series([1.0, 1.1, 1.05, 0.9, 0.95, 1.2])
    # Running max hits 1.1 at index 1; the worst point after that is 0.9 at
    # index 3: 0.9 / 1.1 - 1.
    expected = 0.9 / 1.1 - 1
    assert _max_drawdown(equity) == pytest.approx(expected)


def test_max_drawdown_is_zero_for_a_monotonically_rising_curve():
    equity = pd.Series([1.0, 1.05, 1.1, 1.2])
    assert _max_drawdown(equity) == pytest.approx(0.0)


def test_sharpe_matches_manual_mean_over_std_formula():
    returns = pd.Series([0.01, -0.02, 0.03, 0.015, -0.005])
    periods_per_year = 52
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)
    assert _sharpe(returns, periods_per_year) == pytest.approx(expected)


def test_sharpe_is_zero_for_constant_returns():
    # Zero variance would divide by zero in the raw formula -- should
    # return 0.0 instead of NaN/inf.
    returns = pd.Series([0.01, 0.01, 0.01])
    assert _sharpe(returns, 252) == 0.0


def _synthetic_history(n=300, seed=0, drift=0.0004, vol=0.012):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(log_returns))
    volume = rng.integers(1_000_000, 5_000_000, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


def test_backtest_returns_well_formed_result_on_synthetic_data():
    history = _synthetic_history()
    result = backtest(history, horizon_days=5, train_fraction=0.7)

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.majority_baseline_accuracy <= 1.0
    assert result.n_test_samples > 0
    assert isinstance(result.strategy_total_return, float)
    assert isinstance(result.buy_and_hold_total_return, float)
    assert isinstance(result.beats_baseline, bool)


def test_backtest_raises_with_insufficient_history():
    history = _synthetic_history(n=60)  # not enough after 50-day SMA + split
    with pytest.raises(ValueError):
        backtest(history)


def test_backtest_is_deterministic_for_the_same_input():
    history = _synthetic_history(seed=42)
    result_a = backtest(history)
    result_b = backtest(history)
    assert result_a == result_b
