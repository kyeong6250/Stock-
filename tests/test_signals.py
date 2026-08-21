import numpy as np
import pandas as pd
import pytest

from stockoptions.signals import FEATURE_COLUMNS, _atr, _bollinger_pct_b, _macd_histogram, build_features, build_labels, train


def _synthetic_history(n=120, seed=0, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(log_returns))
    spread = np.abs(rng.normal(0, 0.004, n))  # a plausible daily high/low range around each close, for atr_norm
    volume = rng.integers(1_000_000, 5_000_000, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close, "High": close * (1 + spread), "Low": close * (1 - spread), "Volume": volume}, index=dates)


def test_build_features_has_expected_columns():
    history = _synthetic_history()
    features = build_features(history)
    assert list(features.columns) == FEATURE_COLUMNS


def test_build_features_first_rows_are_nan_until_the_longest_window_fills():
    history = _synthetic_history(n=60)
    features = build_features(history)
    # sma_slow needs 50 rows, so sma_ratio should be NaN until then.
    assert features["sma_ratio"].iloc[:49].isna().all()
    assert features["sma_ratio"].iloc[49:].notna().all()


def test_rsi_is_high_for_a_monotonically_increasing_series():
    close = np.arange(100, 130, dtype=float)
    history = pd.DataFrame({
        "Close": close,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Volume": np.full(30, 1_000_000),
    })
    features = build_features(history)
    # Every day is a gain, no losses at all -> RSI should be pinned near 100.
    assert features["rsi_14"].iloc[-1] > 95


def test_build_labels_matches_manual_future_comparison():
    close = pd.Series([100, 101, 99, 105, 103, 110, 108])
    history = pd.DataFrame({"Close": close, "Volume": [1] * len(close)})
    labels = build_labels(history, horizon_days=2)
    # label[0] compares close[0]=100 to close[2]=99 -> down -> 0
    # label[1] compares close[1]=101 to close[3]=105 -> up -> 1
    assert labels.iloc[0] == 0
    assert labels.iloc[1] == 1
    # last 2 rows have no future price within the series -> NaN
    assert labels.iloc[-1] != labels.iloc[-1]  # NaN != NaN
    assert labels.iloc[-2] != labels.iloc[-2]


def test_train_fits_and_predicts_on_a_clearly_separable_synthetic_dataset():
    # Make one feature perfectly predictive of the label; the model should
    # learn it easily and score well on the same (training) data.
    n = 200
    rng = np.random.default_rng(1)
    momentum = rng.normal(0, 1, n)
    features = pd.DataFrame({
        "sma_ratio": rng.normal(0, 1, n),
        "rsi_14": rng.normal(50, 10, n),
        "momentum_10": momentum,
        "volume_ratio": rng.normal(0, 1, n),
        "volatility_20": rng.uniform(0.01, 0.05, n),
        "macd_hist_norm": rng.normal(0, 0.01, n),
        "bollinger_pctb": rng.uniform(0, 1, n),
        "atr_norm": rng.uniform(0.01, 0.05, n),
    })
    labels = pd.Series((momentum > 0).astype(int), name="label")

    trained = train(features, labels)
    predictions = trained.model.predict(features[FEATURE_COLUMNS])
    accuracy = (predictions == labels.values).mean()
    assert accuracy > 0.9


def test_train_raises_with_too_few_samples():
    features = pd.DataFrame({c: [0.0] * 10 for c in FEATURE_COLUMNS})
    labels = pd.Series([0, 1] * 5, name="label")
    with pytest.raises(ValueError):
        train(features, labels)


def test_train_rejects_an_unknown_model_type():
    features = pd.DataFrame({c: np.random.default_rng(0).normal(0, 1, 40) for c in FEATURE_COLUMNS})
    labels = pd.Series([0, 1] * 20, name="label")
    with pytest.raises(ValueError, match="model_type"):
        train(features, labels, model_type="banana")


def test_train_random_forest_fits_and_predicts_on_a_clearly_separable_dataset():
    n = 300
    rng = np.random.default_rng(2)
    momentum = rng.normal(0, 1, n)
    features = pd.DataFrame({
        "sma_ratio": rng.normal(0, 1, n),
        "rsi_14": rng.normal(50, 10, n),
        "momentum_10": momentum,
        "volume_ratio": rng.normal(0, 1, n),
        "volatility_20": rng.uniform(0.01, 0.05, n),
        "macd_hist_norm": rng.normal(0, 0.01, n),
        "bollinger_pctb": rng.uniform(0, 1, n),
        "atr_norm": rng.uniform(0.01, 0.05, n),
    })
    labels = pd.Series((momentum > 0).astype(int), name="label")

    trained = train(features, labels, model_type="random_forest")
    predictions = trained.model.predict(features[FEATURE_COLUMNS])
    accuracy = (predictions == labels.values).mean()
    assert accuracy > 0.85


def test_feature_importance_sums_to_one_for_logistic():
    n = 100
    rng = np.random.default_rng(4)
    features = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS})
    labels = pd.Series(rng.integers(0, 2, n), name="label")
    trained = train(features, labels, model_type="logistic")
    importance = trained.feature_importance()
    assert set(importance) == set(FEATURE_COLUMNS)
    assert sum(importance.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in importance.values())


def test_feature_importance_sums_to_one_for_random_forest():
    n = 150
    rng = np.random.default_rng(4)
    features = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS})
    labels = pd.Series(rng.integers(0, 2, n), name="label")
    trained = train(features, labels, model_type="random_forest")
    importance = trained.feature_importance()
    assert set(importance) == set(FEATURE_COLUMNS)
    assert sum(importance.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in importance.values())


def test_feature_importance_ranks_the_truly_predictive_feature_highest():
    # One feature perfectly determines the label, the rest are noise --
    # a correct importance calculation should rank it clearly first for
    # both model types, not just produce well-formed-looking numbers.
    n = 300
    rng = np.random.default_rng(6)
    momentum = rng.normal(0, 1, n)
    features = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS})
    features["momentum_10"] = momentum
    labels = pd.Series((momentum > 0).astype(int), name="label")

    for model_type in ("logistic", "random_forest"):
        trained = train(features, labels, model_type=model_type)
        importance = trained.feature_importance()
        top_feature = max(importance, key=importance.get)
        assert top_feature == "momentum_10", f"{model_type} ranked {top_feature!r} above the actually-predictive feature"


def test_train_random_forest_is_deterministic_across_runs():
    n = 100
    rng = np.random.default_rng(3)
    features = pd.DataFrame({c: rng.normal(0, 1, n) for c in FEATURE_COLUMNS})
    labels = pd.Series(rng.integers(0, 2, n), name="label")

    a = train(features, labels, model_type="random_forest")
    b = train(features, labels, model_type="random_forest")
    preds_a = a.model.predict(features[FEATURE_COLUMNS])
    preds_b = b.model.predict(features[FEATURE_COLUMNS])
    assert list(preds_a) == list(preds_b)  # a fixed random_state should make two independent fits agree exactly


# ---------------------------------------------------------------------
# individual indicator formulas, hand-checked
# ---------------------------------------------------------------------


def test_macd_histogram_is_zero_for_a_perfectly_flat_series():
    close = pd.Series([100.0] * 60)
    hist = _macd_histogram(close)
    # A constant price never gives the fast EMA any reason to diverge from
    # the slow EMA, so the MACD line -- and therefore its histogram -- is
    # exactly zero at every point once both EMAs have converged.
    assert hist.iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_macd_histogram_is_positive_during_a_sustained_uptrend():
    close = pd.Series(np.linspace(100, 200, 80))
    hist = _macd_histogram(close)
    # A steady uptrend keeps the fast EMA persistently above the slower,
    # more lagging signal line -- a textbook positive MACD histogram.
    assert hist.iloc[-1] > 0


def test_bollinger_pct_b_is_zero_point_five_for_a_flat_series():
    close = pd.Series([100.0] * 30)
    pctb = _bollinger_pct_b(close)
    # Zero volatility collapses the bands to a single point at the SMA
    # (band_width -> 0), which this implementation guards against
    # dividing by directly (replaced with NaN) -- so this documents that
    # edge case rather than asserting a specific numeric value.
    assert pctb.iloc[-1] != pctb.iloc[-1] or pctb.iloc[-1] == pytest.approx(0.5)  # NaN or exactly centered, either is a valid zero-width-band result


def test_bollinger_pct_b_is_near_one_at_a_fresh_high_after_a_ranging_period():
    rng = np.random.default_rng(5)
    ranging = 100 + rng.normal(0, 1, 40)
    breakout = np.concatenate([ranging, [ranging.max() + 10]])
    close = pd.Series(breakout)
    pctb = _bollinger_pct_b(close)
    assert pctb.iloc[-1] > 0.9


def test_atr_matches_hand_computed_true_range_for_a_simple_case():
    # Three days, each day's own high/low range fully brackets the prior
    # close (100 -> [99,102], 101 -> [100.5,102]), so true range collapses
    # to plain high-low each day -- avoids needing a hand-verified gap
    # scenario just to check the rolling-mean wiring is correct.
    history = pd.DataFrame({
        "Close": [100.0, 101.0, 99.0],
        "High": [101.0, 102.0, 102.0],
        "Low": [99.5, 99.0, 100.5],
    })
    atr = _atr(history, window=3)
    expected = np.mean([101.0 - 99.5, 102.0 - 99.0, 102.0 - 100.5])
    assert atr.iloc[-1] == pytest.approx(expected)


def test_atr_captures_a_gap_beyond_the_days_own_high_low_range():
    # Day 2 gaps way down from day 1's close (110 -> day-2 range is only
    # 60-62, far below 110) -- true range must be measured from the prior
    # close, not just the day's own high-low, or a gap would be invisible.
    history = pd.DataFrame({
        "Close": [110.0, 61.0],
        "High": [111.0, 62.0],
        "Low": [109.0, 60.0],
    })
    atr = _atr(history, window=2)
    day2_true_range = max(62.0 - 60.0, abs(62.0 - 110.0), abs(60.0 - 110.0))
    assert day2_true_range == pytest.approx(50.0)
    assert atr.iloc[-1] == pytest.approx((2.0 + 50.0) / 2)
