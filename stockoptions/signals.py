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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# The original five, plus three added after researching what other
# open-source technical-indicator classifiers use (see the README's
# Accuracy upgrades section for the comparison this led to): MACD and
# Bollinger %B specifically came up repeatedly, and one project's own
# reported random-forest feature-importance ranking called out Bollinger
# Bands as substantially more informative than its other features.
FEATURE_COLUMNS = [
    "sma_ratio",
    "rsi_14",
    "momentum_10",
    "volume_ratio",
    "volatility_20",
    "macd_hist_norm",
    "bollinger_pctb",
    "atr_norm",
]


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


def _macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Standard MACD (12/26/9 EMA periods): the histogram (MACD line minus
    its own signal line) rather than the raw MACD line, since the
    histogram is the momentum-of-momentum reading traders actually watch
    (crossing zero = the trend's momentum just flipped). Divided by close
    in build_features() below to make it scale-invariant across tickers
    with very different price levels, consistent with how sma_ratio and
    momentum_10 are already relative rather than absolute quantities."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _bollinger_pct_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """%B: where price sits within its own Bollinger Bands, as a fraction
    (0 = at the lower band, 1 = at the upper band, naturally bounded
    under normal conditions but can exceed [0, 1] on a genuine breakout).
    Already scale-invariant by construction, unlike a raw band width."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = (upper - lower).replace(0, np.nan)
    return (close - lower) / band_width


def _atr(history: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range: the standard volatility-of-range indicator,
    distinct from volatility_20's close-to-close return volatility since
    it also captures intraday high/low range, not just where each day
    closed. True range = max(high-low, |high-prev_close|, |low-prev_close|)."""
    high, low, close = history["High"], history["Low"], history["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Eight technical indicators from OHLCV history. The first ~50 rows
    will be NaN (the SMA-50/rolling windows need that much history) --
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
    features["macd_hist_norm"] = _macd_histogram(close) / close
    features["bollinger_pctb"] = _bollinger_pct_b(close)
    features["atr_norm"] = _atr(history, 14) / close
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
    model: Pipeline  # StandardScaler -> LogisticRegression or RandomForestClassifier, see train()'s model_type
    feature_columns: list[str]

    def predict_proba_up(self, features_row: pd.Series) -> float:
        X = features_row[self.feature_columns].to_frame().T
        return float(self.model.predict_proba(X)[0][1])


def train(features: pd.DataFrame, labels: pd.Series, model_type: str = "logistic") -> TrainedModel:
    """Fits a StandardScaler + classifier pipeline. Scaling matters here
    specifically because the eight features live on very different scales
    (rsi_14 spans 0-100; sma_ratio and volatility_20 are small decimals)
    -- logistic regression's regularization penalizes large coefficients
    uniformly, so on unscaled features it implicitly under-weights the
    smaller-magnitude ones for reasons that have nothing to do with their
    actual predictive value. Using a Pipeline (rather than scaling
    features separately before calling this function) keeps the same
    fitted scaler applied consistently at both train and predict time --
    there's no way to accidentally score new data with a differently-
    fitted scaler.

    `model_type`: "logistic" (default -- simple, fast, and what every
    number in this project's README was originally measured against, so
    it stays the default rather than being silently swapped out) or
    "random_forest" (multiple sources researched while adding this
    consistently report gradient-boosted/random-forest ensembles
    modestly outperforming plain logistic regression on this kind of
    tabular technical-indicator classification -- see the README's
    Accuracy upgrades section for the actual head-to-head comparison this
    project ran before trusting that claim for its own data, rather than
    just taking it on faith). Scaling is a no-op for tree ensembles
    (RandomForestClassifier is scale-invariant) but kept in the pipeline
    anyway so both model types share one code path and one TrainedModel
    interface. max_depth/min_samples_leaf are deliberately conservative:
    with a few hundred to a couple thousand training rows, an
    unconstrained forest overfits noise easily."""
    data = features.join(labels.rename("label")).dropna()
    if len(data) < 30:
        raise ValueError(f"need at least 30 labeled samples to train, got {len(data)}")
    X = data[FEATURE_COLUMNS]
    y = data["label"].astype(int)

    if model_type == "logistic":
        estimator = LogisticRegression(max_iter=1000)
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=15, random_state=42)
    else:
        raise ValueError(f"model_type must be 'logistic' or 'random_forest', got {model_type!r}")

    model = Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
    model.fit(X, y)
    return TrainedModel(model, FEATURE_COLUMNS)
