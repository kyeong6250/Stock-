import math
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockoptions.blackscholes import call_price, put_price
from stockoptions.recommend import (
    ConePoint,
    RecommendationError,
    estimate_kelly_edge,
    expected_move_cone,
    pick_contract,
    predict_live_direction,
    recommend_trade,
)


def _synthetic_history(n=400, seed=0, drift=0.0004, vol=0.012):
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(log_returns))
    spread = np.abs(rng.normal(0, 0.004, n))  # plausible daily high/low range around each close, for atr_norm
    volume = rng.integers(1_000_000, 5_000_000, n)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close, "High": close * (1 + spread), "Low": close * (1 - spread), "Volume": volume}, index=dates)


# ---------------------------------------------------------------------
# predict_live_direction
# ---------------------------------------------------------------------


def test_predict_live_direction_returns_a_valid_direction_and_probability():
    history = _synthetic_history()
    direction, probability = predict_live_direction(history, horizon_days=5)
    assert direction in ("up", "down")
    assert 0.5 <= probability <= 1.0  # probability is always of the CHOSEN direction, so >= 0.5 by construction


def test_predict_live_direction_raises_with_insufficient_history():
    history = _synthetic_history(n=40)
    with pytest.raises(RecommendationError):
        predict_live_direction(history, horizon_days=5)


def test_predict_live_direction_is_deterministic():
    history = _synthetic_history(seed=42)
    a = predict_live_direction(history, horizon_days=5)
    b = predict_live_direction(history, horizon_days=5)
    assert a == b


# ---------------------------------------------------------------------
# pick_contract
# ---------------------------------------------------------------------


def _synthetic_chain(S, expiration, r, sigma, q=0.0, strikes=None):
    strikes = strikes or [S * m for m in (0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)]
    from stockoptions.data import years_to_expiration

    T = years_to_expiration(expiration)
    rows_calls, rows_puts = [], []
    for K in strikes:
        cp = call_price(S, K, T, r, sigma, q)
        pp = put_price(S, K, T, r, sigma, q)
        rows_calls.append({"strike": K, "bid": cp - 0.01, "ask": cp + 0.01, "lastPrice": cp})
        rows_puts.append({"strike": K, "bid": pp - 0.01, "ask": pp + 0.01, "lastPrice": pp})
    return pd.DataFrame(rows_calls), pd.DataFrame(rows_puts)


def test_pick_contract_selects_the_expiration_closest_to_target_dte():
    S, sigma, r = 100.0, 0.30, 0.04
    near = (date.today() + timedelta(days=10)).isoformat()
    target = (date.today() + timedelta(days=36)).isoformat()
    far = (date.today() + timedelta(days=90)).isoformat()

    chains = {
        near: _synthetic_chain(S, near, r, sigma),
        target: _synthetic_chain(S, target, r, sigma),
        far: _synthetic_chain(S, far, r, sigma),
    }

    with (
        patch("stockoptions.recommend.get_option_expirations", return_value=[near, target, far]),
        patch("stockoptions.recommend.get_option_chain", side_effect=lambda ticker, exp: chains[exp]),
    ):
        contract = pick_contract("SYN", "call", S, q=0.0, target_dte=35, target_delta=0.35, yield_curve={1.0: r})

    assert contract.expiration == target


def test_pick_contract_selects_the_strike_closest_to_target_delta():
    S, sigma, r = 100.0, 0.30, 0.04
    expiration = (date.today() + timedelta(days=35)).isoformat()
    chain_calls, chain_puts = _synthetic_chain(S, expiration, r, sigma)

    with (
        patch("stockoptions.recommend.get_option_expirations", return_value=[expiration]),
        patch("stockoptions.recommend.get_option_chain", return_value=(chain_calls, chain_puts)),
    ):
        contract = pick_contract("SYN", "call", S, q=0.0, target_dte=35, target_delta=0.30, yield_curve={1.0: r})

    assert abs(contract.delta - 0.30) < 0.15  # a real delta-targeting result, not necessarily exact given discrete strikes
    assert contract.option_type == "call"
    assert contract.premium > 0


def test_pick_contract_raises_when_no_contract_has_a_valid_quote():
    expiration = (date.today() + timedelta(days=35)).isoformat()
    bad_chain = pd.DataFrame([{"strike": 100.0, "bid": 0.0, "ask": 0.0, "lastPrice": 0.0}])

    with (
        patch("stockoptions.recommend.get_option_expirations", return_value=[expiration]),
        patch("stockoptions.recommend.get_option_chain", return_value=(bad_chain, bad_chain)),
    ):
        with pytest.raises(RecommendationError, match="no usable"):
            pick_contract("SYN", "call", 100.0, q=0.0, target_dte=35, target_delta=0.35, yield_curve={1.0: 0.04})


# ---------------------------------------------------------------------
# estimate_kelly_edge
# ---------------------------------------------------------------------


def test_estimate_kelly_edge_flags_insufficient_sample_when_few_rows_match_the_predicted_class():
    # Mocking backtest_rows() directly rather than hunting for a history
    # length that happens to land in the narrow gap between "too short
    # for backtest_rows's own minimum" and "enough matching rows to clear
    # MIN_KELLY_SAMPLE" -- that gap turned out to be only 1-2 rows wide
    # for realistic synthetic series, too fragile to depend on.
    from stockoptions.backtest import BacktestRows

    dates = pd.date_range("2023-01-01", periods=8, freq="B")
    fake_rows = BacktestRows(
        train_data=pd.DataFrame({"label": [1.0] * 40}),
        test_data=pd.DataFrame({"Close": np.full(8, 100.0)}, index=dates),
        predictions=np.array([1, 1, 1, 1, 1, 1, 1, 1]),  # all predicted "up" -- fewer than MIN_KELLY_SAMPLE=10 rows total
        forward_returns=pd.Series(np.full(8, 0.01), index=dates),
    )
    with patch("stockoptions.recommend.backtest_rows", return_value=fake_rows):
        edge = estimate_kelly_edge(_synthetic_history(), "call", moneyness=1.05, iv=0.3, r=0.04, q=0.0, horizon_days=5, predicted_direction="up")
    assert edge.insufficient_sample
    assert edge.sample_size == 8
    assert edge.kelly_fraction == 0.0


def test_estimate_kelly_edge_is_positive_for_a_strongly_trending_synthetic_series():
    # Strong, low-noise upward drift: a directional classifier should
    # reliably call "up" correctly on this, and a call option bought
    # whenever it does should show a real, positive empirical edge --
    # exercising the "Kelly correctly finds a real edge when one exists"
    # path, as a complement to the insufficient-sample and (elsewhere)
    # real-ticker "no edge" cases.
    history = _synthetic_history(n=500, seed=1, drift=0.003, vol=0.004)
    direction, _ = predict_live_direction(history, horizon_days=5)
    assert direction == "up"  # sanity check the synthetic series itself is unambiguous before trusting the edge below

    edge = estimate_kelly_edge(history, "call", moneyness=1.0, iv=0.25, r=0.04, q=0.0, horizon_days=5, predicted_direction="up")
    assert not edge.insufficient_sample
    assert edge.win_rate > 0.5
    assert edge.kelly_fraction > 0.0


# ---------------------------------------------------------------------
# expected_move_cone
# ---------------------------------------------------------------------


def test_expected_move_cone_matches_the_standard_formula():
    S, iv = 100.0, 0.30
    points = expected_move_cone(S, iv, max_days=10, step=5)
    assert [p.day for p in points] == [5, 10]
    expected_move_5d = S * iv * math.sqrt(5 / 365)
    assert points[0].upper_1sd == pytest.approx(S + expected_move_5d)
    assert points[0].lower_1sd == pytest.approx(S - expected_move_5d)
    assert points[0].upper_2sd == pytest.approx(S + 2 * expected_move_5d)


def test_expected_move_cone_widens_as_days_increase():
    points = expected_move_cone(100.0, 0.30, max_days=30, step=10)
    widths = [p.upper_1sd - p.lower_1sd for p in points]
    assert widths == sorted(widths)
    assert widths[0] < widths[-1]


def test_expected_move_cone_floors_lower_band_above_zero():
    # A huge IV over a long horizon could otherwise drive the lower band negative.
    points = expected_move_cone(10.0, 3.0, max_days=365, step=365)
    assert all(p.lower_1sd >= 0.01 and p.lower_2sd >= 0.01 for p in points)


def test_expected_move_cone_rejects_non_positive_max_days():
    with pytest.raises(ValueError):
        expected_move_cone(100.0, 0.3, max_days=0)


# ---------------------------------------------------------------------
# recommend_trade (full pipeline, everything mocked)
# ---------------------------------------------------------------------


def _patch_recommend_dependencies(S=100.0, sigma=0.30, r=0.04, q=0.0, history=None):
    expiration = (date.today() + timedelta(days=35)).isoformat()
    calls, puts = _synthetic_chain(S, expiration, r, sigma, q)
    history = history if history is not None else _synthetic_history(n=500, seed=3)
    return [
        patch("stockoptions.recommend.get_price_history", return_value=history),
        patch("stockoptions.recommend.get_current_price", return_value=S),
        patch("stockoptions.recommend.get_dividend_yield", return_value=q),
        patch("stockoptions.recommend.get_yield_curve", return_value={1.0: r}),
        patch("stockoptions.recommend.get_option_expirations", return_value=[expiration]),
        patch("stockoptions.recommend.get_option_chain", return_value=(calls, puts)),
    ]


def test_recommend_trade_returns_a_well_formed_recommendation():
    patches = _patch_recommend_dependencies()
    for p in patches:
        p.start()
    try:
        rec = recommend_trade("SYN", account_size=50_000, max_risk_pct=0.02, horizon_days=35, target_delta=0.35)
    finally:
        for p in patches:
            p.stop()

    assert rec.ticker == "SYN"
    assert rec.direction in ("up", "down")
    assert rec.contract.option_type == ("call" if rec.direction == "up" else "put")
    assert rec.recommended_contracts >= 0
    assert rec.actual_dollar_risk == pytest.approx(rec.recommended_contracts * rec.contract.premium * 100)
    assert rec.actual_dollar_risk <= rec.recommended_dollar_risk + 1e-6
    assert len(rec.cone) == rec.contract.dte
    assert all(isinstance(p, ConePoint) for p in rec.cone)


def test_recommend_trade_never_recommends_more_dollar_risk_than_the_cap_allows():
    patches = _patch_recommend_dependencies()
    for p in patches:
        p.start()
    try:
        rec = recommend_trade("SYN", account_size=1_000_000, max_risk_pct=0.01, horizon_days=35, target_delta=0.35)
    finally:
        for p in patches:
            p.stop()
    assert rec.recommended_dollar_risk == pytest.approx(1_000_000 * 0.01)
    assert rec.actual_dollar_risk <= rec.recommended_dollar_risk + 1e-6


def test_recommend_trade_warns_when_it_does_not_beat_baseline():
    # A pure random walk (zero drift) is exactly the case backtest.py's
    # own README results demonstrate: no real directional edge, so the
    # majority-baseline warning should fire.
    rng = np.random.default_rng(9)
    log_returns = rng.normal(0.0, 0.015, 500)
    close = 100 * np.exp(np.cumsum(log_returns))
    spread = np.abs(rng.normal(0, 0.004, 500))
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    flat_history = pd.DataFrame(
        {"Close": close, "High": close * (1 + spread), "Low": close * (1 - spread), "Volume": rng.integers(1_000_000, 5_000_000, 500)}, index=dates
    )

    patches = _patch_recommend_dependencies(history=flat_history)
    for p in patches:
        p.start()
    try:
        rec = recommend_trade("SYN", account_size=50_000, max_risk_pct=0.02, horizon_days=35, target_delta=0.35)
    finally:
        for p in patches:
            p.stop()
    # Not asserting the warning always fires (a random walk can occasionally
    # still edge out its own baseline by chance) -- just that the mechanism
    # is wired correctly: whenever beats_baseline is False, the warning is present.
    if not rec.backtest.beats_baseline:
        assert any("did not beat" in w for w in rec.warnings)


def test_recommend_trade_rejects_non_positive_account_size():
    with pytest.raises(RecommendationError):
        recommend_trade("SYN", account_size=0)


def test_recommend_trade_rejects_an_out_of_range_max_risk_pct():
    with pytest.raises(RecommendationError):
        recommend_trade("SYN", max_risk_pct=1.5)
