import os

import pytest

from stockoptions.rates import get_yield_curve, risk_free_rate

LIVE = os.environ.get("STOCKOPTIONS_LIVE_TESTS") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE, reason="set STOCKOPTIONS_LIVE_TESTS=1 to run tests that hit the real yfinance API")

SAMPLE_CURVE = {13 / 52: 0.037, 5.0: 0.0435, 10.0: 0.0465, 30.0: 0.0519}


def test_risk_free_rate_at_a_known_maturity_returns_that_exact_yield():
    assert risk_free_rate(5.0, SAMPLE_CURVE) == pytest.approx(0.0435)


def test_risk_free_rate_interpolates_between_two_maturities():
    # Halfway between the 5y (4.35%) and 10y (4.65%) points -> ~4.50%.
    rate = risk_free_rate(7.5, SAMPLE_CURVE)
    assert rate == pytest.approx((0.0435 + 0.0465) / 2, abs=1e-4)


def test_risk_free_rate_clamps_below_the_shortest_maturity():
    # A 1-week option shouldn't extrapolate past the 13-week bill rate.
    assert risk_free_rate(1 / 52, SAMPLE_CURVE) == pytest.approx(0.037)


def test_risk_free_rate_clamps_above_the_longest_maturity():
    assert risk_free_rate(50.0, SAMPLE_CURVE) == pytest.approx(0.0519)


def test_risk_free_rate_rejects_non_positive_T():
    with pytest.raises(ValueError):
        risk_free_rate(0, SAMPLE_CURVE)


@skip_unless_live
def test_get_yield_curve_returns_plausible_real_rates():
    curve = get_yield_curve()
    assert len(curve) > 0
    for maturity, rate in curve.items():
        assert maturity > 0
        # A sanity range wide enough to not be a real assertion about
        # markets, just a check that we got a rate and not, say, a raw
        # unconverted percentage (37.0 instead of 0.37) or garbage.
        assert 0 < rate < 0.25
