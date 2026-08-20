"""Chronological (no-lookahead) train/test backtest for the directional
signal, always reported next to two baselines it has to beat to mean
anything:

- majority-class baseline: the accuracy you'd get by always guessing
  whichever label was more common in the training data. Markets drift
  upward over multi-year periods, so "always guess up" already scores
  well above 50% -- that's not the model finding an edge, it's market
  drift, so accuracy has to clear THIS bar, not a naive 50%, to mean
  anything at all.
- buy-and-hold total return over the same test period.

Known simplification: the "strategy" below opens a new `horizon_days`
position on every day the model says "up," so consecutive trades overlap
rather than modeling realistic non-overlapping position sizing/capital
allocation. Good enough to sanity-check whether the signal has any edge
at all; not a realistic execution simulator.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockoptions.signals import FEATURE_COLUMNS, build_features, build_labels, train


@dataclass
class BacktestResult:
    accuracy: float
    majority_baseline_accuracy: float
    n_test_samples: int
    strategy_total_return: float
    buy_and_hold_total_return: float
    strategy_sharpe: float
    strategy_max_drawdown: float

    @property
    def beats_baseline(self) -> bool:
        return self.accuracy > self.majority_baseline_accuracy


def _sharpe(returns: pd.Series, periods_per_year: float) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def backtest(history: pd.DataFrame, horizon_days: int = 5, train_fraction: float = 0.7) -> BacktestResult:
    features = build_features(history)
    labels = build_labels(history, horizon_days)
    data = features.join(labels).join(history["Close"]).dropna()

    split_idx = int(len(data) * train_fraction)
    if split_idx < 30 or len(data) - split_idx < 10:
        raise ValueError(
            f"not enough history for a meaningful train/test split: {len(data)} usable rows "
            f"after feature/label windows, need at least ~57 (30 train + 10 test minimum)"
        )

    train_data, test_data = data.iloc[:split_idx], data.iloc[split_idx:]

    model = train(train_data[FEATURE_COLUMNS], train_data["label"])
    predictions = model.model.predict(test_data[FEATURE_COLUMNS])

    accuracy = float((predictions == test_data["label"].values).mean())
    majority_label = train_data["label"].mode().iloc[0]
    majority_baseline_accuracy = float((test_data["label"] == majority_label).mean())

    forward_returns = test_data["Close"].shift(-horizon_days) / test_data["Close"] - 1
    strategy_returns = pd.Series(np.where(predictions == 1, forward_returns, 0.0), index=test_data.index).dropna()

    equity_curve = (1 + strategy_returns).cumprod()
    strategy_total_return = float(equity_curve.iloc[-1] - 1) if len(equity_curve) else 0.0
    buy_and_hold_total_return = float(test_data["Close"].iloc[-1] / test_data["Close"].iloc[0] - 1)

    periods_per_year = 252 / horizon_days
    sharpe = _sharpe(strategy_returns, periods_per_year)
    max_dd = _max_drawdown(equity_curve) if len(equity_curve) else 0.0

    return BacktestResult(
        accuracy=accuracy,
        majority_baseline_accuracy=majority_baseline_accuracy,
        n_test_samples=len(test_data),
        strategy_total_return=strategy_total_return,
        buy_and_hold_total_return=buy_and_hold_total_return,
        strategy_sharpe=sharpe,
        strategy_max_drawdown=max_dd,
    )
