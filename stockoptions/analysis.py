"""Shared analysis helpers used by both the CLI and the web dashboard, so
the two surfaces read from one tested implementation instead of drifting
out of sync with each other over time.
"""

from dataclasses import dataclass

from stockoptions.data import (
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
    target = min(expirations, key=lambda e: abs(years_to_expiration(e) - 30 / 365))
    T = years_to_expiration(target)
    curve = yield_curve if yield_curve is not None else get_yield_curve()
    r = risk_free_rate(T, curve)
    q = get_dividend_yield(ticker)

    calls, _ = get_option_chain(ticker, target)
    enriched = enrich_chain_with_iv_and_greeks(calls, S, target, "call", r, q)
    atm_row = enriched.iloc[(enriched["strike"] - S).abs().argsort()[:1]]
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
