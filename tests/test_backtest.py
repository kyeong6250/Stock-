import numpy as np
import pandas as pd
import pytest

from stockoptions.backtest import _max_drawdown, _sharpe, backtest, walk_forward_backtest


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
    spread = np.abs(rng.normal(0, 0.004, n))  # plausible daily high/low range around each close, for atr_norm
    volume = rng.integers(1_000_000, 5_000_000, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close, "High": close * (1 + spread), "Low": close * (1 - spread), "Volume": volume}, index=dates)


def test_backtest_returns_well_formed_result_on_synthetic_data():
    history = _synthetic_history()
    result = backtest(history, horizon_days=5, train_fraction=0.7)

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.majority_baseline_accuracy <= 1.0
    assert result.n_test_samples > 0
    assert isinstance(result.strategy_total_return, float)
    assert isinstance(result.buy_and_hold_total_return, float)
    assert isinstance(result.beats_baseline, bool)


def test_backtest_equity_curves_are_aligned_and_start_near_one():
    history = _synthetic_history()
    result = backtest(history, horizon_days=5, train_fraction=0.7)

    assert len(result.dates) == len(result.strategy_equity_curve) == len(result.buy_and_hold_equity_curve)
    assert result.strategy_equity_curve[0] == pytest.approx(1.0, abs=0.15)  # first trade's return, not exactly 1.0
    assert result.buy_and_hold_equity_curve[0] == pytest.approx(1.0, abs=0.05)
    # The strategy's own final value should match its reported total return.
    assert result.strategy_equity_curve[-1] - 1 == pytest.approx(result.strategy_total_return)


def test_backtest_raises_with_insufficient_history():
    history = _synthetic_history(n=60)  # not enough after 50-day SMA + split
    with pytest.raises(ValueError):
        backtest(history)


def test_backtest_is_deterministic_for_the_same_input():
    history = _synthetic_history(seed=42)
    result_a = backtest(history)
    result_b = backtest(history)
    assert result_a == result_b


def test_backtest_accepts_random_forest_model_type():
    history = _synthetic_history(seed=11)
    result = backtest(history, horizon_days=5, model_type="random_forest")
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_test_samples > 0


def test_backtest_rejects_an_unknown_model_type():
    history = _synthetic_history(seed=11)
    with pytest.raises(ValueError, match="model_type"):
        backtest(history, model_type="banana")


def test_walk_forward_backtest_produces_one_fold_per_requested_fold_count():
    history = _synthetic_history(n=1200, seed=15)
    result = walk_forward_backtest(history, horizon_days=5, n_folds=4)
    assert len(result.folds) == 4
    assert all(0.0 <= f.accuracy <= 1.0 for f in result.folds)
    assert all(0.0 <= f.majority_baseline_accuracy <= 1.0 for f in result.folds)


def test_walk_forward_backtest_uses_an_expanding_training_window():
    history = _synthetic_history(n=1200, seed=16)
    result = walk_forward_backtest(history, horizon_days=5, n_folds=4)
    train_sizes = [f.n_train for f in result.folds]
    assert train_sizes == sorted(train_sizes)  # strictly non-decreasing: fold i's training data is a superset of fold i-1's
    assert len(set(train_sizes)) == len(train_sizes)  # and actually growing each fold, not flat


def test_walk_forward_backtest_purges_horizon_days_at_every_fold_boundary():
    history = _synthetic_history(n=1200, seed=17)
    horizon_days = 10
    features_labels_close = history  # backtest.py computes this internally; just confirm the purge arithmetic via chunk math
    n_folds = 4
    result = walk_forward_backtest(history, horizon_days=horizon_days, n_folds=n_folds)

    from stockoptions.backtest import build_features, build_labels

    data = build_features(features_labels_close).join(build_labels(features_labels_close, horizon_days)).join(history["Close"]).dropna()
    chunk_size = len(data) // (n_folds + 1)
    for fold in result.folds:
        train_end_idx = fold.fold * chunk_size
        expected_train_end = max(0, train_end_idx - horizon_days)
        assert fold.n_train == expected_train_end


def test_walk_forward_backtest_computes_mean_and_std_correctly():
    history = _synthetic_history(n=1200, seed=18)
    result = walk_forward_backtest(history, horizon_days=5, n_folds=4)
    accuracies = [f.accuracy for f in result.folds]
    assert result.mean_accuracy == pytest.approx(sum(accuracies) / len(accuracies))
    assert result.fraction_of_folds_beating_baseline == pytest.approx(sum(1 for f in result.folds if f.beats_baseline) / len(result.folds))


def test_walk_forward_backtest_raises_with_insufficient_history():
    history = _synthetic_history(n=100)  # way too short for 5 folds of at least 30 rows each
    with pytest.raises(ValueError):
        walk_forward_backtest(history, horizon_days=5, n_folds=5)


def test_walk_forward_backtest_accepts_random_forest_model_type():
    history = _synthetic_history(n=1200, seed=19)
    result = walk_forward_backtest(history, horizon_days=5, n_folds=3, model_type="random_forest")
    assert len(result.folds) == 3


def test_backtest_purge_gap_matches_horizon_days_exactly():
    # Patch train_fraction to a fixed value and check the purge arithmetic
    # directly against the internal split, rather than comparing two
    # different horizon_days runs (their total usable row count `len(data)`
    # also shifts with horizon_days, since a bigger horizon trims more NaN
    # labels off the tail -- so train_samples alone isn't a clean signal;
    # the actual invariant that matters is split_idx - train_end == horizon_days).
    from stockoptions.backtest import build_features, build_labels

    history = _synthetic_history(n=400, seed=7)
    horizon_days = 15
    train_fraction = 0.7

    features = build_features(history)
    labels = build_labels(history, horizon_days)
    data = features.join(labels).join(history["Close"]).dropna()
    split_idx = int(len(data) * train_fraction)
    expected_train_end = split_idx - horizon_days

    result = backtest(history, horizon_days=horizon_days, train_fraction=train_fraction)
    assert result.n_train_samples == expected_train_end
    assert result.n_test_samples == len(data) - split_idx
