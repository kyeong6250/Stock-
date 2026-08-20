import numpy as np
import pandas as pd
import pytest

from stockoptions.signals import FEATURE_COLUMNS, build_features, build_labels, train


def _synthetic_history(n=120, seed=0, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(log_returns))
    volume = rng.integers(1_000_000, 5_000_000, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


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
    history = pd.DataFrame({
        "Close": np.arange(100, 130, dtype=float),
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
