"""Shared analysis helpers used by both the CLI and the web dashboard, so
the two surfaces read from one tested implementation instead of drifting
out of sync with each other over time.
"""

from dataclasses import dataclass
from datetime import date

from stockoptions.data import (
    TickerNotFoundError,
    enrich_chain_with_iv_and_greeks,
    get_current_price,
    get_dividend_yield,
    get_option_chain,
    get_option_expirations,
    get_price_history,
    years_to_expiration,
)
from stockoptions.rates import get_yield_curve, risk_free_rate
from stockoptions.volatility import historical_volatility


@dataclass
class TickerOverview:
    ticker: str
    price: float
    atm_iv: float
    realized_vol_30d: float
    iv_hv_ratio: float
    read: str  # "rich" | "cheap" | "in line"
    risk_free_rate: float
    dividend_yield: float
    nearest_expiration: str


def screen_ticker(ticker: str, yield_curve: dict[float, float] | None = None) -> TickerOverview:
    """One ticker's IV-vs-realized-vol read, using a real maturity-matched
    risk-free rate and real dividend yield (see rates.py/data.py). Pass a
    pre-fetched `yield_curve` when screening several tickers in one go to
    avoid refetching the Treasury curve for each one."""
    S = get_current_price(ticker)
    history = get_price_history(ticker, period="3mo")
    hv = historical_volatility(history["Close"], window=30)

    expirations = get_option_expirations(ticker)
    # Same-day (0DTE) expirations -- common for liquid weeklies -- have no
    # usable time-to-expiration (years_to_expiration requires T > 0) and
    # can't be priced. Live-caught: without this filter, min()'s key
    # function raises the moment it evaluates a same-day date, crashing
    # the whole overview/`screen`/`watchlist` path on any day a ticker
    # happens to have one in its expirations list -- not a hypothetical,
    # this is exactly what happened testing this against AAPL's real chain.
    future_expirations = [e for e in expirations if date.fromisoformat(e) > date.today()]
    if not future_expirations:
        raise TickerNotFoundError(f"no future option expirations available for ticker {ticker!r} (only same-day/expired dates found)")
    target = min(future_expirations, key=lambda e: abs(years_to_expiration(e) - 30 / 365))
    T = years_to_expiration(target)
    curve = yield_curve if yield_curve is not None else get_yield_curve()
    r = risk_free_rate(T, curve)
    q = get_dividend_yield(ticker)

    calls, _ = get_option_chain(ticker, target)
    enriched = enrich_chain_with_iv_and_greeks(calls, S, target, "call", r, q)
    # Filter to rows with a valid recomputed IV *before* picking the
    # closest-to-spot strike, not after: enrich_chain_with_iv_and_greeks
    # already sets my_iv to NaN for any contract with a bad/stale quote
    # (see its own docstring), and the literal nearest strike to spot is
    # exactly the kind of near-the-money contract that can have thin,
    # crossed, or zero quotes on an illiquid name -- picking it blindly
    # would either silently carry a NaN IV into the rich/cheap read below
    # (a wrong verdict, not a crash) or IndexError outright if the whole
    # chain came back with no valid quotes at all.
    valid = enriched.dropna(subset=["my_iv"])
    if valid.empty:
        raise TickerNotFoundError(f"no option contract with a usable quote found for {ticker!r} at expiration {target}")
    atm_row = valid.iloc[(valid["strike"] - S).abs().argsort()[:1]]
    iv = float(atm_row["my_iv"].iloc[0])

    ratio = iv / hv if hv else float("nan")
    read = "rich" if ratio > 1.15 else ("cheap" if ratio < 0.85 else "in line")

    return TickerOverview(
        ticker=ticker,
        price=S,
        atm_iv=iv,
        realized_vol_30d=hv,
        iv_hv_ratio=ratio,
        read=read,
        risk_free_rate=r,
        dividend_yield=q,
        nearest_expiration=target,
    )
