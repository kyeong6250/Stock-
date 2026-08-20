"""Multi-leg options strategy payoff analysis: P&L at expiration, max
profit/loss, and breakevens for any combination of calls, puts, and stock.

Unlike the volatility/signal modules, nothing here is a guess -- given the
legs of a position, its payoff at expiration is exact arithmetic, not a
prediction. This is deliberately the most directly useful piece of the
whole project for actually evaluating a trade before you place it.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.optimize import brentq


class Instrument(Enum):
    CALL = "call"
    PUT = "put"
    STOCK = "stock"


@dataclass(frozen=True)
class Leg:
    """One leg of a position. `quantity` is signed: positive = long,
    negative = short. `premium` is the price paid (long) or received
    (short) per share for an option leg, or the entry price per share for
    a stock leg. `strike` is ignored (must be None) for stock legs."""

    instrument: Instrument
    quantity: int
    strike: float | None = None
    premium: float = 0.0

    def __post_init__(self):
        if self.quantity == 0:
            raise ValueError("a leg with quantity=0 isn't a position")
        if self.instrument == Instrument.STOCK:
            if self.strike is not None:
                raise ValueError("stock legs don't have a strike")
        elif self.strike is None or self.strike <= 0:
            raise ValueError(f"{self.instrument.value} legs need a positive strike")

    def payoff(self, S_T: float) -> float:
        """P&L per share at expiration, for the underlying at S_T."""
        if self.instrument == Instrument.CALL:
            intrinsic = max(S_T - self.strike, 0.0)
        elif self.instrument == Instrument.PUT:
            intrinsic = max(self.strike - S_T, 0.0)
        else:
            intrinsic = S_T
        return self.quantity * (intrinsic - self.premium)


@dataclass(frozen=True)
class Strategy:
    name: str
    legs: list[Leg] = field(default_factory=list)

    def payoff(self, S_T: float) -> float:
        return sum(leg.payoff(S_T) for leg in self.legs)

    def net_premium(self) -> float:
        """Positive = net debit paid to open. Negative = net credit received."""
        return sum(leg.quantity * leg.premium for leg in self.legs)

    def _upside_slope(self) -> float:
        """The payoff's slope as S_T -> infinity: puts flatten out (their
        intrinsic value goes to 0), so only call and stock legs matter."""
        return sum(leg.quantity for leg in self.legs if leg.instrument in (Instrument.CALL, Instrument.STOCK))


def payoff_curve(strategy: Strategy, low: float, high: float, num: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(low, high, num)
    ys = np.array([strategy.payoff(x) for x in xs])
    return xs, ys


def default_scan_range(strategy: Strategy) -> tuple[float, float]:
    """A sane default price range to scan: 0 up to 2x the highest strike
    involved (or 2x the max premium-implied scale if there are no strikes,
    e.g. a stock-only position)."""
    strikes = [leg.strike for leg in strategy.legs if leg.strike is not None]
    if strikes:
        return 0.0, 2 * max(strikes)
    premiums = [leg.premium for leg in strategy.legs if leg.premium > 0]
    scale = max(premiums) if premiums else 100.0
    return 0.0, 4 * scale


def max_profit(strategy: Strategy, low: float | None = None, high: float | None = None) -> float:
    """Returns float('inf') if profit is genuinely unbounded (e.g. a naked
    long call/stock), rather than a misleading finite number from
    whatever scan range happened to be used."""
    if strategy._upside_slope() > 0:
        return float("inf")
    lo, hi = default_scan_range(strategy) if low is None or high is None else (low, high)
    _, ys = payoff_curve(strategy, lo, hi)
    return float(ys.max())


def max_loss(strategy: Strategy, low: float | None = None, high: float | None = None) -> float:
    """Returns float('-inf') if loss is genuinely unbounded (e.g. a naked
    short call). Downside is always naturally bounded by S_T=0 (a stock
    price can't go negative), so only the upside needs this check."""
    if strategy._upside_slope() < 0:
        return float("-inf")
    lo, hi = default_scan_range(strategy) if low is None or high is None else (low, high)
    _, ys = payoff_curve(strategy, lo, hi)
    return float(ys.min())


def breakevens(strategy: Strategy, low: float | None = None, high: float | None = None, num: int = 4000) -> list[float]:
    """Underlying prices at expiration where P&L crosses zero, found by
    scanning for sign changes and root-finding within each one."""
    lo, hi = default_scan_range(strategy) if low is None or high is None else (low, high)
    xs, ys = payoff_curve(strategy, lo, hi, num)
    roots = []
    for i in range(len(xs) - 1):
        if ys[i] == 0:
            roots.append(float(xs[i]))
        elif ys[i] * ys[i + 1] < 0:
            root = brentq(strategy.payoff, xs[i], xs[i + 1])
            roots.append(float(root))
    return roots


# --- Convenience constructors for common strategies ---


def long_call(strike: float, premium: float, quantity: int = 1) -> Strategy:
    return Strategy("Long Call", [Leg(Instrument.CALL, quantity, strike, premium)])


def long_put(strike: float, premium: float, quantity: int = 1) -> Strategy:
    return Strategy("Long Put", [Leg(Instrument.PUT, quantity, strike, premium)])


def covered_call(stock_entry: float, call_strike: float, call_premium: float, quantity: int = 1) -> Strategy:
    # Both legs use share-equivalent units (100 shares per contract) so they
    # net out correctly in the upside-slope check that max_profit/max_loss
    # use to detect unbounded payoffs -- a stock leg of +100*quantity shares
    # against a call leg of only -quantity would look like it still has net
    # positive slope (unbounded profit) when the call in fact caps it.
    return Strategy(
        "Covered Call",
        [
            Leg(Instrument.STOCK, 100 * quantity, premium=stock_entry),
            Leg(Instrument.CALL, -100 * quantity, call_strike, call_premium),
        ],
    )


def vertical_spread(
    option_type: str, long_strike: float, short_strike: float, long_premium: float, short_premium: float, quantity: int = 1
) -> Strategy:
    """A bull call spread (option_type='call', long_strike < short_strike)
    or a bear put spread (option_type='put', long_strike > short_strike) --
    buy one strike, sell another, same expiration and type."""
    instrument = Instrument.CALL if option_type == "call" else Instrument.PUT
    return Strategy(
        f"{option_type.title()} Vertical Spread",
        [
            Leg(instrument, quantity, long_strike, long_premium),
            Leg(instrument, -quantity, short_strike, short_premium),
        ],
    )


def strangle(
    call_strike: float, put_strike: float, call_premium: float, put_premium: float, quantity: int = 1, short: bool = True
) -> Strategy:
    """Short strangle (sell OTM call + OTM put, collect premium, bet on low
    realized volatility) or long strangle (buy both, bet on a big move)."""
    sign = -quantity if short else quantity
    label = "Short Strangle" if short else "Long Strangle"
    return Strategy(
        label,
        [
            Leg(Instrument.CALL, sign, call_strike, call_premium),
            Leg(Instrument.PUT, sign, put_strike, put_premium),
        ],
    )


def straddle(strike: float, call_premium: float, put_premium: float, quantity: int = 1, short: bool = True) -> Strategy:
    sign = -quantity if short else quantity
    label = "Short Straddle" if short else "Long Straddle"
    return Strategy(
        label,
        [
            Leg(Instrument.CALL, sign, strike, call_premium),
            Leg(Instrument.PUT, sign, strike, put_premium),
        ],
    )


def iron_condor(
    put_long_strike: float,
    put_short_strike: float,
    call_short_strike: float,
    call_long_strike: float,
    put_long_premium: float,
    put_short_premium: float,
    call_short_premium: float,
    call_long_premium: float,
    quantity: int = 1,
) -> Strategy:
    """Classic 4-leg iron condor: buy a far OTM put (protection), sell a
    closer OTM put, sell a closer OTM call, buy a far OTM call (protection).
    Strikes must satisfy put_long < put_short < call_short < call_long."""
    if not (put_long_strike < put_short_strike < call_short_strike < call_long_strike):
        raise ValueError("strikes must satisfy put_long < put_short < call_short < call_long")
    return Strategy(
        "Iron Condor",
        [
            Leg(Instrument.PUT, quantity, put_long_strike, put_long_premium),
            Leg(Instrument.PUT, -quantity, put_short_strike, put_short_premium),
            Leg(Instrument.CALL, -quantity, call_short_strike, call_short_premium),
            Leg(Instrument.CALL, quantity, call_long_strike, call_long_premium),
        ],
    )
