import pytest

from stockoptions.strategies import (
    Instrument,
    Leg,
    Strategy,
    breakevens,
    covered_call,
    iron_condor,
    long_call,
    max_loss,
    max_profit,
    strangle,
    vertical_spread,
)


def test_long_call_leg_payoff_itm_and_otm():
    leg = Leg(Instrument.CALL, 1, strike=100, premium=5)
    assert leg.payoff(110) == pytest.approx(5)  # 10 intrinsic - 5 premium
    assert leg.payoff(90) == pytest.approx(-5)  # worthless, lose the premium


def test_short_call_leg_is_the_mirror_of_long():
    long_leg = Leg(Instrument.CALL, 1, strike=100, premium=5)
    short_leg = Leg(Instrument.CALL, -1, strike=100, premium=5)
    for S_T in (80, 100, 120, 200):
        assert short_leg.payoff(S_T) == pytest.approx(-long_leg.payoff(S_T))


def test_long_put_leg_payoff():
    leg = Leg(Instrument.PUT, 1, strike=100, premium=4)
    assert leg.payoff(90) == pytest.approx(6)
    assert leg.payoff(110) == pytest.approx(-4)


def test_stock_leg_payoff_long_and_short():
    long_stock = Leg(Instrument.STOCK, 100, premium=50)
    short_stock = Leg(Instrument.STOCK, -100, premium=50)
    assert long_stock.payoff(55) == pytest.approx(500)
    assert short_stock.payoff(55) == pytest.approx(-500)


@pytest.mark.parametrize("bad_leg", [
    dict(instrument=Instrument.CALL, quantity=0, strike=100),
    dict(instrument=Instrument.CALL, quantity=1, strike=None),
    dict(instrument=Instrument.STOCK, quantity=1, strike=100),
])
def test_leg_validation_rejects_nonsensical_legs(bad_leg):
    with pytest.raises(ValueError):
        Leg(**bad_leg)


def test_long_call_strategy_has_unbounded_profit_and_capped_loss():
    strat = long_call(strike=100, premium=5)
    assert max_profit(strat) == float("inf")
    assert max_loss(strat) == pytest.approx(-5)


def test_naked_short_call_has_unbounded_loss_and_capped_profit():
    strat = Strategy("Naked Short Call", [Leg(Instrument.CALL, -1, 100, 5)])
    assert max_loss(strat) == float("-inf")
    assert max_profit(strat) == pytest.approx(5)


def test_bull_call_spread_max_profit_loss_and_breakeven():
    # Buy the 100 call for 8, sell the 110 call for 3: net debit 5.
    strat = vertical_spread("call", long_strike=100, short_strike=110, long_premium=8, short_premium=3)
    assert strat.net_premium() == pytest.approx(5)  # positive = debit paid
    assert max_profit(strat) == pytest.approx(5)  # (110-100) - 5
    assert max_loss(strat) == pytest.approx(-5)  # -net debit
    assert breakevens(strat) == pytest.approx([105], abs=0.01)


def test_iron_condor_max_profit_loss_and_breakevens():
    strat = iron_condor(
        put_long_strike=90,
        put_short_strike=95,
        call_short_strike=105,
        call_long_strike=110,
        put_long_premium=1,
        put_short_premium=2,
        call_short_premium=2,
        call_long_premium=1,
    )
    assert strat.net_premium() == pytest.approx(-2)  # negative = credit received
    assert max_profit(strat) == pytest.approx(2)  # the credit
    assert max_loss(strat) == pytest.approx(-3)  # wing width (5) - credit (2)
    assert breakevens(strat) == pytest.approx([93, 107], abs=0.01)


def test_short_strangle_has_unbounded_loss_on_the_call_side():
    strat = strangle(call_strike=110, put_strike=90, call_premium=2, put_premium=2, short=True)
    assert max_loss(strat) == float("-inf")  # naked short call component
    assert max_profit(strat) == pytest.approx(4)  # both premiums, if S_T lands between strikes


def test_covered_call_caps_upside_at_the_strike():
    # Own 100 shares at $50, sell the $55 call for $1.50/share.
    strat = covered_call(stock_entry=50, call_strike=55, call_premium=1.5)
    # Max profit: capped at (55-50+1.5)*100 = 650, achieved at or above $55.
    assert max_profit(strat) == pytest.approx(650, abs=0.5)
    # Payoff should be flat (capped) well above the strike, not still rising.
    assert strat.payoff(60) == pytest.approx(strat.payoff(100))
