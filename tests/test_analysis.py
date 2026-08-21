import os
from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockoptions.analysis import TickerOverview, screen_ticker
from stockoptions.blackscholes import call_price
from stockoptions.data import TickerNotFoundError

LIVE = os.environ.get("STOCKOPTIONS_LIVE_TESTS") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE, reason="set STOCKOPTIONS_LIVE_TESTS=1 to run tests that hit the real yfinance API")


def _synthetic_call_chain(S, expiration, r, sigma, q=0.0):
    from stockoptions.data import years_to_expiration

    T = years_to_expiration(expiration)
    strikes = [S * m for m in (0.9, 0.95, 1.0, 1.05, 1.1)]
    rows = [{"strike": K, "bid": call_price(S, K, T, r, sigma, q) - 0.01, "ask": call_price(S, K, T, r, sigma, q) + 0.01, "lastPrice": call_price(S, K, T, r, sigma, q)} for K in strikes]
    return pd.DataFrame(rows)


def _synthetic_price_history(n=90, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    dates = pd.date_range(end=date.today().isoformat(), periods=n, freq="B")
    return pd.DataFrame({"Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n)}, index=dates)


def test_screen_ticker_excludes_a_same_day_expiration_from_target_selection():
    # Live-caught bug: a same-day (0DTE) expiration in the list used to
    # crash screen_ticker entirely, since years_to_expiration requires
    # T > 0 and min()'s key function evaluated it mid-comparison rather
    # than it just losing to a better candidate.
    today = date.today().isoformat()
    target_30d = (date.today() + timedelta(days=30)).isoformat()
    far = (date.today() + timedelta(days=90)).isoformat()
    S, sigma, r = 100.0, 0.3, 0.04
    chain = _synthetic_call_chain(S, target_30d, r, sigma)

    with (
        patch("stockoptions.analysis.get_current_price", return_value=S),
        patch("stockoptions.analysis.get_price_history", return_value=_synthetic_price_history()),
        patch("stockoptions.analysis.get_option_expirations", return_value=[today, target_30d, far]),
        patch("stockoptions.analysis.get_option_chain", return_value=(chain, chain)),
        patch("stockoptions.analysis.get_dividend_yield", return_value=0.0),
    ):
        overview = screen_ticker("SYN", yield_curve={1.0: r})

    assert overview.nearest_expiration == target_30d


def test_screen_ticker_raises_a_clear_error_when_only_same_day_expirations_exist():
    today = date.today().isoformat()
    with (
        patch("stockoptions.analysis.get_current_price", return_value=100.0),
        patch("stockoptions.analysis.get_price_history", return_value=_synthetic_price_history()),
        patch("stockoptions.analysis.get_option_expirations", return_value=[today]),
    ):
        with pytest.raises(TickerNotFoundError, match="no future option expirations"):
            screen_ticker("SYN", yield_curve={1.0: 0.04})


@skip_unless_live
def test_screen_ticker_returns_a_well_formed_overview():
    overview = screen_ticker("AAPL")
    assert isinstance(overview, TickerOverview)
    assert overview.ticker == "AAPL"
    assert overview.price > 0
    assert overview.atm_iv > 0
    assert overview.realized_vol_30d > 0
    assert overview.read in ("rich", "cheap", "in line")
    assert 0 < overview.risk_free_rate < 0.25
    assert overview.dividend_yield >= 0
    assert overview.nearest_expiration


@skip_unless_live
def test_screen_ticker_reuses_a_provided_yield_curve():
    from stockoptions.rates import get_yield_curve

    curve = get_yield_curve()
    overview = screen_ticker("AAPL", yield_curve=curve)
    assert overview.risk_free_rate in curve.values() or 0 < overview.risk_free_rate < 0.25
