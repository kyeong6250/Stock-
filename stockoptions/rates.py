"""Real, maturity-matched risk-free rates from the Treasury yield curve,
instead of a single hardcoded constant.

Confirmed while researching this: "the risk-free rate is taken as the
yield on a risk-free security... with a maturity that matches the time to
expiration of the option," interpolating between available maturities
when the option's term falls between two Treasury tenors (see the
README's Accuracy upgrades section for sources). A flat guessed rate
(the project used to default to r=0.05 everywhere) doesn't reflect that
at all -- short and long rates can differ by more than a percentage
point, live-checked while building this (13-week bill at 3.70% vs
30-year bond at 5.19% is a real example seen during development).
"""

import numpy as np
import yfinance as yf

# yfinance tickers for on-the-run Treasury yields and their maturity in
# years. These tickers quote yield directly as a percentage (e.g. a
# lastPrice of 4.53 means 4.53%), confirmed live against several tickers
# while building this -- not assumed.
_TREASURY_TICKERS: dict[str, float] = {
    "^IRX": 13 / 52,  # 13-week T-bill
    "^FVX": 5.0,  # 5-year note
    "^TNX": 10.0,  # 10-year note
    "^TYX": 30.0,  # 30-year bond
}


def get_yield_curve() -> dict[float, float]:
    """{maturity_in_years: annualized yield as a decimal}, from current
    Treasury quotes. Raises RuntimeError if none of the tickers return
    usable data (e.g. markets closed with stale/missing quotes)."""
    curve: dict[float, float] = {}
    for ticker, maturity in _TREASURY_TICKERS.items():
        last_price = yf.Ticker(ticker).fast_info.get("lastPrice")
        if last_price:
            curve[maturity] = float(last_price) / 100
    if not curve:
        raise RuntimeError("could not fetch any Treasury yield data (yfinance returned nothing usable)")
    return curve


def risk_free_rate(T: float, curve: dict[float, float] | None = None) -> float:
    """A risk-free rate matched to time-to-expiration T (years), linearly
    interpolated between the two closest known Treasury maturities (or
    clamped to the nearest end if T is outside the curve's range -- e.g.
    an option expiring in a week uses the 13-week bill rate rather than
    extrapolating past it)."""
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    curve = curve if curve is not None else get_yield_curve()
    maturities = sorted(curve)
    if T <= maturities[0]:
        return curve[maturities[0]]
    if T >= maturities[-1]:
        return curve[maturities[-1]]
    return float(np.interp(T, maturities, [curve[m] for m in maturities]))
