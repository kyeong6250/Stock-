import math
import os
from datetime import date

import pandas as pd
import pytest

from stockoptions.blackscholes import call_price, greeks
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

LIVE = os.environ.get("STOCKOPTIONS_LIVE_TESTS") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE, reason="set STOCKOPTIONS_LIVE_TESTS=1 to run tests that hit the real yfinance API")


def test_years_to_expiration_computes_day_count_over_365():
    result = years_to_expiration("2026-12-31", as_of=date(2026, 1, 1))
    expected_days = (date(2026, 12, 31) - date(2026, 1, 1)).days
    assert result == pytest.approx(expected_days / 365.0)


def test_years_to_expiration_rejects_a_non_future_date():
    with pytest.raises(ValueError):
        years_to_expiration("2020-01-01", as_of=date(2026, 1, 1))


def test_mid_price_prefers_bid_ask_midpoint():
    row = pd.Series({"bid": 4.0, "ask": 5.0, "lastPrice": 100.0})
    assert _mid_price(row) == pytest.approx(4.5)


def test_mid_price_falls_back_to_last_price_when_no_bid_ask():
    row = pd.Series({"bid": 0.0, "ask": 0.0, "lastPrice": 3.2})
    assert _mid_price(row) == pytest.approx(3.2)


def test_mid_price_returns_none_when_nothing_usable():
    row = pd.Series({"bid": 0.0, "ask": 0.0, "lastPrice": 0.0})
    assert _mid_price(row) is None


def test_enrich_chain_recovers_a_known_iv_black_scholes_model():
    S, K, r, sigma = 100.0, 100.0, 0.05, 0.28
    exp_str = date.today().replace(year=date.today().year + 1).isoformat()  # safely in the future
    T = years_to_expiration(exp_str)  # use the exact T enrich() will compute internally

    price = call_price(S, K, T, r, sigma)
    chain = pd.DataFrame([{"strike": K, "bid": price - 0.01, "ask": price + 0.01, "lastPrice": price}])
    enriched = enrich_chain_with_iv_and_greeks(chain, S, exp_str, "call", r, pricing_model="black-scholes")

    assert enriched["my_iv"].iloc[0] == pytest.approx(sigma, abs=1e-4)
    expected_greeks = greeks(S, K, T, r, sigma, "call")
    assert enriched["delta"].iloc[0] == pytest.approx(expected_greeks.delta, abs=1e-6)


def test_enrich_chain_recovers_a_known_iv_binomial_model_default():
    from stockoptions import binomial

    S, K, r, sigma = 100.0, 95.0, 0.04, 0.30
    exp_str = date.today().replace(year=date.today().year + 1).isoformat()
    T = years_to_expiration(exp_str)

    target_price = binomial.price(S, K, T, r, sigma, "put")  # put: early-exercise value actually differs from B-S here
    chain = pd.DataFrame([{"strike": K, "bid": target_price - 0.01, "ask": target_price + 0.01, "lastPrice": target_price}])
    enriched = enrich_chain_with_iv_and_greeks(chain, S, exp_str, "put", r)  # default pricing_model="binomial"

    assert enriched["my_iv"].iloc[0] == pytest.approx(sigma, abs=1e-3)


def test_enrich_chain_rejects_an_unknown_pricing_model():
    exp_str = date.today().replace(year=date.today().year + 1).isoformat()
    chain = pd.DataFrame([{"strike": 100.0, "bid": 5.0, "ask": 5.2, "lastPrice": 5.1}])
    with pytest.raises(ValueError):
        enrich_chain_with_iv_and_greeks(chain, 100.0, exp_str, "call", pricing_model="banana")


def test_enrich_chain_gives_nan_for_an_unusable_quote():
    exp_str = date.today().replace(year=date.today().year + 1).isoformat()
    chain = pd.DataFrame([{"strike": 100.0, "bid": 0.0, "ask": 0.0, "lastPrice": 0.0}])
    enriched = enrich_chain_with_iv_and_greeks(chain, 100.0, exp_str, "call")
    assert math.isnan(enriched["my_iv"].iloc[0])


def test_enrich_chain_gives_nan_for_a_no_arbitrage_violation_rather_than_raising():
    # A call priced above S is impossible under Black-Scholes -- should be
    # skipped as NaN, not crash the whole batch.
    exp_str = date.today().replace(year=date.today().year + 1).isoformat()
    chain = pd.DataFrame([{"strike": 100.0, "bid": 150.0, "ask": 151.0, "lastPrice": 150.5}])
    enriched = enrich_chain_with_iv_and_greeks(chain, 100.0, exp_str, "call")
    assert math.isnan(enriched["my_iv"].iloc[0])


@skip_unless_live
def test_get_price_history_returns_real_data():
    df = get_price_history("AAPL", period="5d")
    assert not df.empty
    assert "Close" in df.columns


@skip_unless_live
def test_get_current_price_returns_a_positive_number():
    assert get_current_price("AAPL") > 0


@skip_unless_live
def test_get_option_expirations_returns_dates():
    expirations = get_option_expirations("AAPL")
    assert len(expirations) > 0


@skip_unless_live
def test_get_option_chain_has_expected_columns():
    expirations = get_option_expirations("AAPL")
    calls, puts = get_option_chain("AAPL", expirations[0])
    assert "strike" in calls.columns
    assert "strike" in puts.columns


@skip_unless_live
def test_get_dividend_yield_is_plausible_for_a_known_dividend_payer():
    # KO has consistently paid a meaningful dividend for decades -- a
    # sanity range wide enough to not assert anything about markets, just
    # to catch a units bug (e.g. accidentally returning 2.39 instead of
    # 0.0239, a mistake this function's docstring exists specifically to
    # avoid making).
    assert 0.005 < get_dividend_yield("KO") < 0.10


@skip_unless_live
def test_get_dividend_yield_is_zero_for_a_non_dividend_payer():
    assert get_dividend_yield("TSLA") == pytest.approx(0.0)
