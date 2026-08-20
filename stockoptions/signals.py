"""Technical-indicator features and a simple directional classifier.

This is deliberately the weakest, most skeptically-framed piece of the
whole project. Predicting short-term stock direction from price history
alone is a famously hard problem -- markets are close to a random walk at
short horizons, which is *why* backtest.py always reports this model's
accuracy next to a majority-class baseline rather than a bare number.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = ["sma_ratio", "rsi_14", "momentum_10", "volume_ratio", "volatility_20"]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # avg_loss == 0 divides by zero above (now NaN in `rsi`). Resolve those
    # rows explicitly: a run with no losses at all is maximally overbought
    # (100), not neutral -- only genuinely flat prices (no gains AND no
    # losses in the window) are conventionally 50. Rows still NaN because
    # the rolling window hasn't filled yet are left as NaN (avg_loss is
    # NaN there too, so neither mask below matches them).
    no_losses = avg_loss == 0
    rsi[no_losses & (avg_gain > 0)] = 100.0
    rsi[no_losses & (avg_gain == 0)] = 50.0
    return rsi


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Five simple technical indicators from OHLCV history. The first ~50
    rows will be NaN (the SMA-50/rolling windows need that much history) --
    that's expected, drop them (e.g. via .dropna()) before training."""
    close = history["Close"]
    volume = history["Volume"]

    sma_fast = close.rolling(10).mean()
    sma_slow = close.rolling(50).mean()

    features = pd.DataFrame(index=history.index)
    features["sma_ratio"] = sma_fast / sma_slow - 1
    features["rsi_14"] = _rsi(close, 14)
    features["momentum_10"] = close.pct_change(10)
    features["volume_ratio"] = volume / volume.rolling(20).mean() - 1
    features["volatility_20"] = close.pct_change().rolling(20).std()
    return features


def build_labels(history: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    """1 if price is higher `horizon_days` sessions later, else 0. The last
    `horizon_days` rows will be NaN (no future price to compare against yet)."""
    close = history["Close"]
    future = close.shift(-horizon_days)
    label = pd.Series(np.where(future > close, 1, 0), index=history.index, dtype=float)
    label[future.isna()] = np.nan
    return label.rename("label")


@dataclass
class TrainedModel:
    model: LogisticRegression
    feature_columns: list[str]

    def predict_proba_up(self, features_row: pd.Series) -> float:
        X = features_row[self.feature_columns].to_frame().T
        return float(self.model.predict_proba(X)[0][1])


def train(features: pd.DataFrame, labels: pd.Series) -> TrainedModel:
    data = features.join(labels.rename("label")).dropna()
    if len(data) < 30:
        raise ValueError(f"need at least 30 labeled samples to train, got {len(data)}")
    X = data[FEATURE_COLUMNS]
    y = data["label"].astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return TrainedModel(model, FEATURE_COLUMNS)
