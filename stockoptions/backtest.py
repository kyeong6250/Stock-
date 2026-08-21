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
    n_train_samples: int
    n_test_samples: int
    strategy_total_return: float
    buy_and_hold_total_return: float
    strategy_sharpe: float
    strategy_max_drawdown: float
    dates: list[str]  # test-period dates (ISO), for charting the two curves below
    strategy_equity_curve: list[float]  # both start at 1.0
    buy_and_hold_equity_curve: list[float]
    feature_importance: dict[str, float]  # from the TRAINING-window model; see signals.TrainedModel.feature_importance()

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


@dataclass
class BacktestRows:
    """The raw per-day test-period data backtest() aggregates into a
    BacktestResult -- exposed separately so other callers (recommend.py's
    empirical position sizing) can compute their own statistics over the
    same honestly-split, no-leakage test set instead of re-deriving the
    train/test/purge logic themselves and risking it drifting out of sync
    with backtest()'s own version."""

    train_data: pd.DataFrame  # FEATURE_COLUMNS + "label" + "Close", training rows only
    test_data: pd.DataFrame  # same columns, test rows only
    predictions: np.ndarray  # model's 0/1 prediction for each test_data row, same order
    forward_returns: pd.Series  # actual (Close[t+horizon]/Close[t] - 1) for each test_data row
    feature_importance: dict[str, float]  # from the training-window model, see signals.TrainedModel.feature_importance()


def backtest_rows(history: pd.DataFrame, horizon_days: int = 5, train_fraction: float = 0.7, model_type: str = "logistic") -> BacktestRows:
    """Trains on the purged-gap training window and predicts across the
    held-out test window, returning the raw per-row data rather than
    aggregate metrics. See backtest()'s docstring for the purge-gap
    rationale (identical logic, factored out here so both functions share
    one implementation). `model_type` is passed straight through to
    signals.train() -- see its docstring for "logistic" vs "random_forest"."""
    features = build_features(history)
    labels = build_labels(history, horizon_days)
    data = features.join(labels).join(history["Close"]).dropna()

    split_idx = int(len(data) * train_fraction)
    # Purge the last `horizon_days` rows before the split from training:
    # row i's label was computed by looking horizon_days rows ahead, so
    # without this gap, training rows within horizon_days of the boundary
    # have labels that peek across it into the test set -- exactly the
    # "moving average/label computed at the boundary depends on prices
    # that extend into the out-of-sample period" leakage pattern confirmed
    # while researching this (see the README's Accuracy upgrades section).
    train_end = max(0, split_idx - horizon_days)
    if train_end < 30 or len(data) - split_idx < 10:
        raise ValueError(
            f"not enough history for a meaningful train/test split: {len(data)} usable rows "
            f"after feature/label windows and a {horizon_days}-day purge gap, need at least "
            f"~{30 + horizon_days + 10} (30 train + purge gap + 10 test minimum)"
        )

    train_data, test_data = data.iloc[:train_end], data.iloc[split_idx:]

    model = train(train_data[FEATURE_COLUMNS], train_data["label"], model_type=model_type)
    predictions = model.model.predict(test_data[FEATURE_COLUMNS])
    forward_returns = test_data["Close"].shift(-horizon_days) / test_data["Close"] - 1

    return BacktestRows(
        train_data=train_data,
        test_data=test_data,
        predictions=predictions,
        forward_returns=forward_returns,
        feature_importance=model.feature_importance(),
    )


def backtest(history: pd.DataFrame, horizon_days: int = 5, train_fraction: float = 0.7, model_type: str = "logistic") -> BacktestResult:
    rows = backtest_rows(history, horizon_days, train_fraction, model_type=model_type)
    train_data, test_data, predictions, forward_returns = rows.train_data, rows.test_data, rows.predictions, rows.forward_returns

    accuracy = float((predictions == test_data["label"].values).mean())
    majority_label = train_data["label"].mode().iloc[0]
    majority_baseline_accuracy = float((test_data["label"] == majority_label).mean())

    strategy_returns = pd.Series(np.where(predictions == 1, forward_returns, 0.0), index=test_data.index).dropna()

    equity_curve = (1 + strategy_returns).cumprod()
    strategy_total_return = float(equity_curve.iloc[-1] - 1) if len(equity_curve) else 0.0
    buy_and_hold_total_return = float(test_data["Close"].iloc[-1] / test_data["Close"].iloc[0] - 1)

    periods_per_year = 252 / horizon_days
    sharpe = _sharpe(strategy_returns, periods_per_year)
    max_dd = _max_drawdown(equity_curve) if len(equity_curve) else 0.0

    # Buy-and-hold curve over the same dates as the strategy curve (which
    # drops the last horizon_days test rows, no forward return to compute
    # there yet) so the two are directly comparable point-for-point on a chart.
    buy_hold_curve = (test_data["Close"] / test_data["Close"].iloc[0]).loc[strategy_returns.index]

    return BacktestResult(
        accuracy=accuracy,
        majority_baseline_accuracy=majority_baseline_accuracy,
        n_train_samples=len(train_data),
        n_test_samples=len(test_data),
        strategy_total_return=strategy_total_return,
        buy_and_hold_total_return=buy_and_hold_total_return,
        strategy_sharpe=sharpe,
        strategy_max_drawdown=max_dd,
        dates=[d.strftime("%Y-%m-%d") for d in strategy_returns.index],
        strategy_equity_curve=[float(v) for v in equity_curve],
        buy_and_hold_equity_curve=[float(v) for v in buy_hold_curve],
        feature_importance=rows.feature_importance,
    )


@dataclass
class WalkForwardFold:
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    majority_baseline_accuracy: float

    @property
    def beats_baseline(self) -> bool:
        return self.accuracy > self.majority_baseline_accuracy


@dataclass
class WalkForwardResult:
    """The single train/test split backtest() uses can look better or
    worse than the model's real skill purely because of where that one
    boundary happened to fall -- confirmed while researching this
    project's original leakage fix (see this module's docstring and the
    README's Accuracy upgrades section): walk-forward validation is the
    standard way to check whether a single-split number is representative
    or a lucky/unlucky draw, by re-training and re-testing across several
    chronological windows and reporting the SPREAD, not just one point."""

    folds: list[WalkForwardFold]
    mean_accuracy: float
    std_accuracy: float  # sample std across folds; 0.0 if fewer than 2 folds
    mean_baseline_accuracy: float
    fraction_of_folds_beating_baseline: float


def walk_forward_backtest(
    history: pd.DataFrame, horizon_days: int = 5, n_folds: int = 5, model_type: str = "logistic"
) -> WalkForwardResult:
    """Splits the usable history into `n_folds` + 1 roughly equal
    chronological chunks. For fold i, trains on an *expanding* window
    (every chunk before it, i.e. all history available up to that point --
    not a fixed-size rolling window, since a real trader wouldn't discard
    old data just to keep window size constant) and tests on chunk i,
    purging the last `horizon_days` training rows before each boundary
    exactly like backtest()'s single split does. Each fold gets its own
    majority-class baseline (computed from that fold's own training data,
    since the label balance can drift across a multi-year history) rather
    than reusing one global baseline.

    Meaningfully more expensive than backtest() -- `n_folds` full model
    fits instead of one -- so this is opt-in (`--walk-forward` on the
    CLI), not run by default on every backtest."""
    features = build_features(history)
    labels = build_labels(history, horizon_days)
    data = features.join(labels).join(history["Close"]).dropna()

    n = len(data)
    chunk_size = n // (n_folds + 1)
    if chunk_size < 30:
        raise ValueError(
            f"not enough history for {n_folds} walk-forward folds: {n} usable rows split into "
            f"{n_folds + 1} chunks gives only ~{chunk_size} rows/chunk, need at least 30"
        )

    folds = []
    for i in range(1, n_folds + 1):
        train_end_idx = i * chunk_size
        test_end_idx = (i + 1) * chunk_size if i < n_folds else n
        purge_end = max(0, train_end_idx - horizon_days)

        train_data = data.iloc[:purge_end]
        test_data = data.iloc[train_end_idx:test_end_idx]
        if len(train_data) < 30 or len(test_data) < 5:
            continue

        model = train(train_data[FEATURE_COLUMNS], train_data["label"], model_type=model_type)
        predictions = model.model.predict(test_data[FEATURE_COLUMNS])
        accuracy = float((predictions == test_data["label"].values).mean())
        majority_label = train_data["label"].mode().iloc[0]
        baseline = float((test_data["label"] == majority_label).mean())
        folds.append(WalkForwardFold(fold=i, n_train=len(train_data), n_test=len(test_data), accuracy=accuracy, majority_baseline_accuracy=baseline))

    if not folds:
        raise ValueError("no valid walk-forward folds could be built -- try fewer folds or more history")

    accuracies = [f.accuracy for f in folds]
    baselines = [f.majority_baseline_accuracy for f in folds]
    beats = sum(1 for f in folds if f.beats_baseline)

    return WalkForwardResult(
        folds=folds,
        mean_accuracy=float(np.mean(accuracies)),
        std_accuracy=float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        mean_baseline_accuracy=float(np.mean(baselines)),
        fraction_of_folds_beating_baseline=beats / len(folds),
    )
