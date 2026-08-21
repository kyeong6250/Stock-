"""A concrete trade suggestion: given a ticker, which option to buy, how
many, and a forward price-projection chart -- not just the raw up/down
signal backtest.py reports.

Read backtest.py's docstring first if you haven't: the directional model
this builds on has already been shown, honestly, to *not* reliably beat a
majority-class baseline (see the README's Honest results section). This
module doesn't pretend that changed. What it adds on top:

1. **Contract selection** is real, standard practitioner mechanics, not
   prediction: nearest expiration to a target days-to-expiration (30-45
   DTE is the conventional "theta high, gamma low" sweet spot for a
   multi-week directional hold -- see the README's Trade recommender
   section for sources), and a strike chosen by delta-targeting on the
   real, recomputed chain (enrich_chain_with_iv_and_greeks). This part is
   just options mechanics and would be exactly this reliable regardless
   of whether the directional signal has any edge at all.

2. **Position sizing is a real Kelly-criterion calculation, not a fixed
   guess** -- and specifically an *empirical* one: rather than assuming a
   payout ratio, it replays the exact historical test-period rows
   backtest.py already uses (same purge-gap split, same no-lookahead
   guarantee) through the chosen contract's own moneyness and current IV
   via the binomial pricing model, to see what buying this shape of
   contract would actually have paid off historically whenever the model
   predicted this specific direction. The one disclosed approximation:
   historical implied volatility isn't available from free data, so
   *today's* IV stands in for every historical date's IV in that replay.

Because sizing is real Kelly math fed a real (and often weak, per
backtest.py's own honest numbers) edge estimate, it will frequently and
correctly recommend risking very little or nothing -- Kelly's own math
does that when win-rate/payout-ratio don't justify a bet, not this module
second-guessing itself. See TradeRecommendation.warnings for exactly why,
whenever that happens.

Still not investment advice: number of contracts is a mechanical output
of a documented formula fed a small-sample historical estimate, not a
guarantee anything here will work.
"""

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from stockoptions import binomial
from stockoptions.backtest import BacktestResult, backtest, backtest_rows
from stockoptions.data import (
    _mid_price,
    enrich_chain_with_iv_and_greeks,
    get_current_price,
    get_dividend_yield,
    get_option_chain,
    get_option_expirations,
    get_price_history,
    years_to_expiration,
)
from stockoptions.rates import get_yield_curve, risk_free_rate
from stockoptions.signals import FEATURE_COLUMNS, build_features, build_labels, train

MIN_KELLY_SAMPLE = 10  # fewer historical instances than this and the win-rate/payout estimate is too noisy to trust


class RecommendationError(ValueError):
    """Raised when a recommendation can't be produced (not enough history, no usable chain, etc.)."""


def predict_live_direction(history, horizon_days: int = 35) -> tuple[str, float]:
    """Today's directional call: trains fresh on *all* available labeled
    history (not the purged train/test split backtest() uses -- that
    split exists to produce an honest held-out accuracy estimate, not the
    best live model) and scores the single most recent feature row, which
    has no label yet because the future hasn't happened. Returns
    (direction, probability_of_that_direction), both derived from the
    same predict_proba the classifier actually outputs -- "probability"
    here is the model's own calibration, not a separate confidence score."""
    features = build_features(history)
    labels = build_labels(history, horizon_days)
    labeled = features.join(labels).dropna()
    if len(labeled) < 30:
        raise RecommendationError(f"need at least 30 labeled samples to train a live model, got {len(labeled)}")

    latest = features.iloc[-1]
    if latest.isna().any():
        raise RecommendationError("not enough recent history to compute today's feature row (need the rolling-window warmup)")

    model = train(labeled[FEATURE_COLUMNS], labeled["label"])
    proba_up = model.predict_proba_up(latest)
    direction = "up" if proba_up >= 0.5 else "down"
    probability = proba_up if direction == "up" else 1 - proba_up
    return direction, float(probability)


@dataclass
class SelectedContract:
    expiration: str
    dte: int
    option_type: str  # "call" | "put"
    strike: float
    premium: float
    iv: float
    delta: float
    T: float
    r: float


def pick_contract(
    ticker: str,
    option_type: str,
    S: float,
    q: float,
    target_dte: int,
    target_delta: float,
    yield_curve: dict[float, float] | None = None,
    pricing_model: str = "binomial",
) -> SelectedContract:
    """Nearest-to-target-DTE expiration, then the strike whose recomputed
    |delta| is closest to `target_delta` on that expiration's real,
    recomputed chain -- standard delta-targeting (see README sources).
    Raises RecommendationError if the ticker has no options or every
    contract on the chosen expiration fails to produce a valid IV (same
    "skip bad quotes, don't silently trust them" policy as data.py)."""
    expirations = get_option_expirations(ticker)
    today = date.today()
    expiration = min(expirations, key=lambda e: abs((date.fromisoformat(e) - today).days - target_dte))
    T = years_to_expiration(expiration)
    r = risk_free_rate(T, yield_curve)

    calls, puts = get_option_chain(ticker, expiration)
    chain = calls if option_type == "call" else puts
    enriched = enrich_chain_with_iv_and_greeks(chain, S, expiration, option_type, r, q, pricing_model=pricing_model)
    valid = enriched.dropna(subset=["my_iv"]).copy()
    if valid.empty:
        raise RecommendationError(f"no usable {option_type} contracts found for {ticker} at expiration {expiration} (no valid quotes)")

    valid["delta_gap"] = (valid["delta"].abs() - target_delta).abs()
    best = valid.loc[valid["delta_gap"].idxmin()]
    premium = _mid_price(best)
    if premium is None:
        premium = float(best["lastPrice"])

    return SelectedContract(
        expiration=expiration,
        dte=(date.fromisoformat(expiration) - today).days,
        option_type=option_type,
        strike=float(best["strike"]),
        premium=float(premium),
        iv=float(best["my_iv"]),
        delta=float(best["delta"]),
        T=T,
        r=r,
    )


@dataclass
class KellyEdge:
    sample_size: int
    win_rate: float
    avg_win: float  # dollars per contract-share (i.e. per unit of underlying, not x100) among winning historical replays
    avg_loss: float  # same units, among losing replays (always > 0: a magnitude)
    kelly_fraction: float  # raw Kelly f*, clamped to >= 0 (a negative result just means "don't take this trade")
    insufficient_sample: bool


def estimate_kelly_edge(
    history,
    option_type: str,
    moneyness: float,
    iv: float,
    r: float,
    q: float,
    horizon_days: int,
    predicted_direction: str,
    train_fraction: float = 0.7,
) -> KellyEdge:
    """The empirical edge estimate described in this module's docstring:
    replays backtest.py's own purged-gap test-period rows -- restricted
    to the ones where the model predicted the SAME direction being
    recommended today -- through the binomial model at the chosen
    contract's moneyness (K/S, held constant across the replay so each
    historical instance gets an equivalent contract) and *current* IV
    (the disclosed approximation: no free source of historical IV).

    Deliberately takes `horizon_days`, not a `T` in years, and prices
    every replayed premium at T = horizon_days / 365 -- NOT the real
    selected contract's own (possibly longer) time-to-expiration. Caught
    live while testing this: pricing the hypothetical premium at the
    contract's full T while only replaying a horizon_days-ahead price
    move compares an option's full time value against a move that had
    far less time to develop, understating every replayed payoff and
    making even a genuinely winning direction look like a loser. Using
    horizon_days for both keeps "how much the option cost" and "how far
    the price could realistically move in that same window" consistent
    with each other -- this treats every replayed trade as bought and
    held to expiration exactly horizon_days later, which is the same
    assumption the real recommendation makes by targeting an expiration
    near horizon_days out in the first place."""
    T = horizon_days / 365
    rows = backtest_rows(history, horizon_days, train_fraction)
    predicted_class = 1 if predicted_direction == "up" else 0
    mask = rows.predictions == predicted_class

    entries = rows.test_data["Close"][mask]
    fwd = rows.forward_returns[mask].dropna()
    common_idx = entries.index.intersection(fwd.index)
    entries, fwd = entries.loc[common_idx], fwd.loc[common_idx]

    if len(entries) < MIN_KELLY_SAMPLE:
        return KellyEdge(sample_size=len(entries), win_rate=0.0, avg_win=0.0, avg_loss=0.0, kelly_fraction=0.0, insufficient_sample=True)

    pnls = []
    for S_i, ret_i in zip(entries.to_numpy(), fwd.to_numpy()):
        S_future = S_i * (1 + ret_i)
        K_i = S_i * moneyness
        try:
            premium_i = binomial.price(S_i, K_i, T, r, iv, option_type, q)
        except ValueError:
            continue  # a degenerate historical S_i (e.g. near-zero) producing an unpriceable node -- skip, don't crash the whole estimate
        payoff_i = max(S_future - K_i, 0.0) if option_type == "call" else max(K_i - S_future, 0.0)
        pnls.append(payoff_i - premium_i)

    pnls_arr = np.array(pnls)
    if len(pnls_arr) < MIN_KELLY_SAMPLE:
        return KellyEdge(sample_size=len(pnls_arr), win_rate=0.0, avg_win=0.0, avg_loss=0.0, kelly_fraction=0.0, insufficient_sample=True)

    wins = pnls_arr[pnls_arr > 0]
    losses = -pnls_arr[pnls_arr <= 0]
    win_rate = len(wins) / len(pnls_arr)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0:
        kelly = 0.0
    else:
        b = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / b

    return KellyEdge(
        sample_size=len(pnls_arr),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        kelly_fraction=max(0.0, float(kelly)),
        insufficient_sample=len(pnls_arr) < MIN_KELLY_SAMPLE,
    )


@dataclass
class ConePoint:
    day: int
    upper_1sd: float
    lower_1sd: float
    upper_2sd: float
    lower_2sd: float


def expected_move_cone(S: float, iv: float, max_days: int, step: int = 1) -> list[ConePoint]:
    """The standard "expected move" projection used across options
    platforms (Barchart, tastytrade, projectoption, etc. -- see README
    sources): expected_move(t) = S * IV * sqrt(t / 365), giving a
    1-standard-deviation band (~68% of a lognormal-ish outcome
    distribution under the same constant-volatility assumption
    Black-Scholes already makes everywhere else in this project) and a
    2-standard-deviation band (~95%). This is a probability *range*, not
    a prediction of where the price will land -- the whole point of
    showing it as a widening cone rather than a single line is to make
    that uncertainty visible instead of implying false precision."""
    if max_days < 1:
        raise ValueError(f"max_days must be >= 1, got {max_days}")
    points = []
    for day in range(step, max_days + 1, step):
        move = S * iv * math.sqrt(day / 365)
        points.append(
            ConePoint(
                day=day,
                upper_1sd=S + move,
                lower_1sd=max(0.01, S - move),
                upper_2sd=S + 2 * move,
                lower_2sd=max(0.01, S - 2 * move),
            )
        )
    return points


@dataclass
class TradeRecommendation:
    ticker: str
    price: float
    direction: str  # "up" | "down"
    live_probability: float  # model's own calibrated probability of `direction`, as of today
    backtest: BacktestResult  # honest out-of-sample accuracy/baseline for THIS ticker/horizon, for context
    contract: SelectedContract
    edge: KellyEdge
    kelly_multiplier: float  # e.g. 0.5 for half-Kelly
    max_risk_pct: float  # hard user-specified ceiling, applied regardless of what Kelly says
    recommended_risk_fraction: float  # min(kelly_fraction * kelly_multiplier, max_risk_pct), floored at 0
    recommended_dollar_risk: float  # account_size * recommended_risk_fraction (the budget, before rounding to whole contracts)
    recommended_contracts: int
    actual_dollar_risk: float  # recommended_contracts * premium * 100 (the real amount after rounding down to whole contracts)
    cone: list[ConePoint] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def recommend_trade(
    ticker: str,
    account_size: float = 10_000.0,
    max_risk_pct: float = 0.02,
    horizon_days: int = 35,
    target_delta: float = 0.35,
    kelly_multiplier: float = 0.5,
    pricing_model: str = "binomial",
) -> TradeRecommendation:
    """The single entry point tying everything in this module together.
    `horizon_days` drives three things at once, deliberately kept as one
    knob rather than three: the model's forecast horizon, the target
    days-to-expiration for contract selection, and the holding period
    the Kelly replay assumes -- keeping all three in sync is what makes
    "the model predicts X over this horizon" and "this is the contract
    sized for that horizon" the same claim rather than two that happen to
    be shown next to each other."""
    if account_size <= 0:
        raise RecommendationError(f"account_size must be positive, got {account_size}")
    if not (0 < max_risk_pct <= 1):
        raise RecommendationError(f"max_risk_pct must be in (0, 1], got {max_risk_pct}")

    ticker = ticker.upper()
    history = get_price_history(ticker, period="2y")
    S = get_current_price(ticker)
    q = get_dividend_yield(ticker)
    yield_curve = get_yield_curve()

    direction, probability = predict_live_direction(history, horizon_days)
    bt = backtest(history, horizon_days=horizon_days)

    option_type = "call" if direction == "up" else "put"
    contract = pick_contract(ticker, option_type, S, q, horizon_days, target_delta, yield_curve, pricing_model)
    moneyness = contract.strike / S

    edge = estimate_kelly_edge(history, option_type, moneyness, contract.iv, contract.r, q, horizon_days, direction)

    risk_fraction = max(0.0, min(edge.kelly_fraction * kelly_multiplier, max_risk_pct))
    dollar_budget = account_size * risk_fraction
    cost_per_contract = contract.premium * 100
    num_contracts = int(dollar_budget // cost_per_contract) if cost_per_contract > 0 else 0
    actual_dollar_risk = num_contracts * cost_per_contract

    cone = expected_move_cone(S, contract.iv, contract.dte)

    warnings = []
    if not bt.beats_baseline:
        warnings.append(
            f"The directional signal did not beat its majority-class baseline on {ticker} at this horizon "
            f"in backtesting ({bt.accuracy:.1%} vs {bt.majority_baseline_accuracy:.1%}) -- no demonstrated "
            "statistical edge on this specific ticker/horizon."
        )
    if edge.insufficient_sample:
        warnings.append(
            f"Only {edge.sample_size} historical test-period instance(s) matched this direction -- too few "
            "to trust the win-rate/payout estimate behind the sizing below."
        )
    if edge.kelly_fraction <= 0 and not edge.insufficient_sample:
        warnings.append("Kelly criterion came out at or below zero for this contract: the estimated edge doesn't justify risking capital on it.")
    if num_contracts == 0:
        warnings.append("Recommended size rounds down to 0 contracts at this account size / risk cap.")

    return TradeRecommendation(
        ticker=ticker,
        price=S,
        direction=direction,
        live_probability=probability,
        backtest=bt,
        contract=contract,
        edge=edge,
        kelly_multiplier=kelly_multiplier,
        max_risk_pct=max_risk_pct,
        recommended_risk_fraction=risk_fraction,
        recommended_dollar_risk=dollar_budget,
        recommended_contracts=num_contracts,
        actual_dollar_risk=actual_dollar_risk,
        cone=cone,
        warnings=warnings,
    )
