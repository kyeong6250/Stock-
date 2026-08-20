import pytest

from stockoptions import blackscholes
from stockoptions.binomial import NoArbitrageViolation, implied_volatility, price, price_and_greeks

# Coarser steps for test speed; still fine for the tolerances used here.
STEPS = 120


def test_american_call_on_non_dividend_stock_converges_to_black_scholes():
    # Classic result: it's never optimal to early-exercise an American call
    # on a non-dividend-paying stock, so the American and European prices
    # should coincide (the binomial tree should never actually use its
    # early-exercise branch here).
    # CRR converges to Black-Scholes at O(1/N) (confirmed while researching
    # this module), so a few hundred steps gets close but not exact --
    # loosen the tolerance accordingly rather than chasing an unrealistic
    # match at a step count still fast enough to run in a test suite.
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
    american = price(S, K, T, r, sigma, "call", q=0.0, steps=500)
    european = blackscholes.call_price(S, K, T, r, sigma, q=0.0)
    assert american == pytest.approx(european, rel=5e-3)


def test_american_put_is_worth_at_least_as_much_as_european():
    # Early exercise can be optimal for puts even with no dividend, so the
    # American price should be >= the European price, generally strictly >.
    S, K, T, r, sigma = 100, 110, 1.0, 0.05, 0.25
    american = price(S, K, T, r, sigma, "put", q=0.0, steps=STEPS)
    european = blackscholes.put_price(S, K, T, r, sigma, q=0.0)
    assert american >= european - 1e-6


def test_american_call_with_dividends_is_worth_at_least_as_much_as_european():
    # A large dividend yield can make early exercise of a call optimal too.
    S, K, T, r, sigma, q = 100, 90, 1.0, 0.03, 0.25, 0.06
    american = price(S, K, T, r, sigma, "call", q=q, steps=STEPS)
    european = blackscholes.call_price(S, K, T, r, sigma, q=q)
    assert american >= european - 1e-6


def test_price_rejects_bad_option_type():
    with pytest.raises(ValueError):
        price(100, 100, 1.0, 0.05, 0.2, "banana")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_round_trips(option_type):
    S, K, T, r, sigma, q = 100, 95, 0.5, 0.04, 0.30, 0.01
    target = price(S, K, T, r, sigma, option_type, q, steps=STEPS)
    recovered = implied_volatility(target, S, K, T, r, option_type, q, steps=STEPS)
    assert recovered == pytest.approx(sigma, abs=1e-3)


def test_implied_volatility_rejects_unreachable_price():
    with pytest.raises(NoArbitrageViolation):
        implied_volatility(1000, 100, 100, 1.0, 0.05, "call", steps=STEPS)


def test_call_delta_between_zero_and_one():
    _, g = price_and_greeks(100, 100, 1.0, 0.05, 0.20, "call", steps=STEPS)
    assert 0 < g.delta < 1


def test_put_delta_between_minus_one_and_zero():
    _, g = price_and_greeks(100, 100, 1.0, 0.05, 0.20, "put", steps=STEPS)
    assert -1 < g.delta < 0


def test_gamma_is_positive():
    _, g = price_and_greeks(100, 100, 1.0, 0.05, 0.20, "call", steps=STEPS)
    assert g.gamma > 0


def test_greeks_roughly_match_black_scholes_when_early_exercise_has_no_value():
    # No dividend, ATM call: American == European in value, and the tree-
    # extracted Greeks should roughly match the closed-form ones too.
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
    _, tree_greeks = price_and_greeks(S, K, T, r, sigma, "call", q=0.0, steps=250)
    bs_greeks = blackscholes.greeks(S, K, T, r, sigma, "call", q=0.0)
    assert tree_greeks.delta == pytest.approx(bs_greeks.delta, abs=0.02)
    assert tree_greeks.gamma == pytest.approx(bs_greeks.gamma, rel=0.1)
    assert tree_greeks.vega == pytest.approx(bs_greeks.vega, rel=0.1)
