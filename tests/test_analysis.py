import os

import pytest

from stockoptions.analysis import TickerOverview, screen_ticker

LIVE = os.environ.get("STOCKOPTIONS_LIVE_TESTS") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE, reason="set STOCKOPTIONS_LIVE_TESTS=1 to run tests that hit the real yfinance API")


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
