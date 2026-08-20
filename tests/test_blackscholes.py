import math

import pytest

from stockoptions.blackscholes import (
    NoArbitrageViolation,
    call_price,
    greeks,
    implied_volatility,
    put_price,
)

# A handful of varied, plausible scenarios: (S, K, T, r, sigma, q)
SCENARIOS = [
    (100, 100, 1.0, 0.05, 0.20, 0.0),  # ATM, no dividend
    (100, 100, 0.25, 0.05, 0.20, 0.02),  # ATM, short-dated, with dividend
    (100, 120, 0.5, 0.03, 0.35, 0.0),  # OTM call / ITM put
    (100, 80, 0.5, 0.03, 0.35, 0.0),  # ITM call / OTM put
    (50, 50, 2.0, 0.01, 0.60, 0.0),  # long-dated, high vol
]


@pytest.mark.parametrize("S,K,T,r,sigma,q", SCENARIOS)
def test_put_call_parity_holds(S, K, T, r, sigma, q):
    c = call_price(S, K, T, r, sigma, q)
    p = put_price(S, K, T, r, sigma, q)
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert c - p == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("S,K,T,r,sigma,q", SCENARIOS)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_volatility_round_trips(S, K, T, r, sigma, q, option_type):
    pricer = call_price if option_type == "call" else put_price
    price = pricer(S, K, T, r, sigma, q)
    recovered = implied_volatility(price, S, K, T, r, option_type, q)
    assert recovered == pytest.approx(sigma, abs=1e-6)


def test_implied_volatility_rejects_price_below_intrinsic():
    # A call struck deep ITM priced below its own intrinsic value is a
    # no-arbitrage violation -- no volatility could produce it.
    S, K, T, r = 100, 50, 1.0, 0.05
    intrinsic = S - K * math.exp(-r * T)
    with pytest.raises(NoArbitrageViolation):
        implied_volatility(intrinsic - 1, S, K, T, r, "call")


def test_implied_volatility_rejects_price_above_upper_bound():
    S, K, T, r = 100, 100, 1.0, 0.05
    with pytest.raises(NoArbitrageViolation):
        implied_volatility(S + 1, S, K, T, r, "call")  # a call can never be worth more than S


def test_call_delta_between_zero_and_one():
    g = greeks(100, 100, 1.0, 0.05, 0.20, "call")
    assert 0 < g.delta < 1


def test_put_delta_between_minus_one_and_zero():
    g = greeks(100, 100, 1.0, 0.05, 0.20, "put")
    assert -1 < g.delta < 0


def test_deep_itm_call_delta_approaches_one():
    g = greeks(100, 1, 1.0, 0.05, 0.20, "call")
    assert g.delta > 0.999


def test_deep_otm_call_delta_approaches_zero():
    g = greeks(100, 100_000, 1.0, 0.05, 0.20, "call")
    assert g.delta < 0.001


def test_gamma_and_vega_are_positive_and_shared_between_call_and_put():
    call_g = greeks(100, 100, 1.0, 0.05, 0.20, "call")
    put_g = greeks(100, 100, 1.0, 0.05, 0.20, "put")
    assert call_g.gamma > 0
    assert call_g.vega > 0
    assert call_g.gamma == pytest.approx(put_g.gamma)
    assert call_g.vega == pytest.approx(put_g.vega)


def test_delta_put_call_parity():
    # d(Call)/dS - d(Put)/dS = e^{-qT}, the derivative form of put-call parity.
    q = 0.02
    T = 0.5
    call_g = greeks(100, 90, T, 0.05, 0.25, "call", q)
    put_g = greeks(100, 90, T, 0.05, 0.25, "put", q)
    assert call_g.delta - put_g.delta == pytest.approx(math.exp(-q * T), abs=1e-9)


@pytest.mark.parametrize("bad_kwargs", [
    dict(S=0, K=100, T=1, r=0.05, sigma=0.2),
    dict(S=100, K=0, T=1, r=0.05, sigma=0.2),
    dict(S=100, K=100, T=0, r=0.05, sigma=0.2),
    dict(S=100, K=100, T=1, r=0.05, sigma=0),
])
def test_call_price_rejects_non_positive_inputs(bad_kwargs):
    with pytest.raises(ValueError):
        call_price(**bad_kwargs)


def test_greeks_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        greeks(100, 100, 1.0, 0.05, 0.2, "banana")
