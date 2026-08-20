"""yfinance data access with simple local caching (QOL: avoid re-hitting
the network -- and yfinance's rate limits -- on every run within a short
session), plus enriching a raw option chain with our own recomputed IV
and Greeks.

We deliberately don't trust yfinance's own `impliedVolatility` column for
anything IV-rank/skew related: it's often wildly wrong for illiquid,
stale-quoted, or deep ITM/OTM contracts (trivially checkable -- pull a
deep ITM chain and you'll see values like 800% IV that are obviously
noise, not a real market view). We instead recompute IV ourselves from
each contract's own mid-price via blackscholes.implied_volatility(), and
skip (as NaN) any contract whose price doesn't correspond to a valid IV
rather than letting one bad row poison the whole batch.
"""

import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from stockoptions.blackscholes import NoArbitrageViolation, greeks, implied_volatility

CACHE_DIR = Path.home() / ".cache" / "stockoptions"
DEFAULT_TTL_SECONDS = 15 * 60


class TickerNotFoundError(ValueError):
    """Raised when yfinance has no data at all for a ticker (typo, delisted, etc.)."""


def _cache_file(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _cache_fresh(path: Path, ttl: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl


def get_price_history(ticker: str, period: str = "1y", ttl: int = DEFAULT_TTL_SECONDS) -> pd.DataFrame:
    """Daily OHLCV history, cached locally for `ttl` seconds."""
    cache_path = _cache_file(f"{ticker}_history_{period}.csv")
    if _cache_fresh(cache_path, ttl):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise TickerNotFoundError(f"no price history found for ticker {ticker!r}")
    df.to_csv(cache_path)
    return df


def get_current_price(ticker: str) -> float:
    info = yf.Ticker(ticker).fast_info
    price = info.get("lastPrice") if hasattr(info, "get") else getattr(info, "last_price", None)
    if not price:
        raise TickerNotFoundError(f"no current price available for ticker {ticker!r}")
    return float(price)


def get_option_expirations(ticker: str) -> list[str]:
    expirations = yf.Ticker(ticker).options
    if not expirations:
        raise TickerNotFoundError(f"no options chain available for ticker {ticker!r} (ETF/index or no listed options)")
    return list(expirations)


def get_option_chain(ticker: str, expiration: str, ttl: int = DEFAULT_TTL_SECONDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (calls, puts) DataFrames for one expiration date (YYYY-MM-DD)."""
    calls_path = _cache_file(f"{ticker}_{expiration}_calls.csv")
    puts_path = _cache_file(f"{ticker}_{expiration}_puts.csv")
    if _cache_fresh(calls_path, ttl) and _cache_fresh(puts_path, ttl):
        return pd.read_csv(calls_path), pd.read_csv(puts_path)

    chain = yf.Ticker(ticker).option_chain(expiration)
    chain.calls.to_csv(calls_path, index=False)
    chain.puts.to_csv(puts_path, index=False)
    return chain.calls, chain.puts


def years_to_expiration(expiration: str, as_of: date | None = None) -> float:
    exp_date = date.fromisoformat(expiration)
    today = as_of or date.today()
    days = (exp_date - today).days
    if days <= 0:
        raise ValueError(f"expiration {expiration} is not in the future (as of {today})")
    return days / 365.0


def _mid_price(row: pd.Series) -> float | None:
    bid, ask, last = row.get("bid", 0), row.get("ask", 0), row.get("lastPrice", 0)
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2
    if last and last > 0:
        return float(last)
    return None


def enrich_chain_with_iv_and_greeks(
    chain: pd.DataFrame, S: float, expiration: str, option_type: str, r: float = 0.05, q: float = 0.0
) -> pd.DataFrame:
    """Returns a copy of `chain` with my_iv/delta/gamma/vega/theta/rho
    columns added, derived from each contract's own mid-price rather than
    trusted from yfinance's impliedVolatility column. Rows whose price
    doesn't correspond to any valid Black-Scholes IV (stale/zero quotes,
    crossed markets, etc.) get NaN in the new columns instead of raising --
    one bad contract in a chain of fifty shouldn't kill the whole screen."""
    T = years_to_expiration(expiration)
    out = chain.copy()
    ivs, deltas, gammas, vegas, thetas, rhos = [], [], [], [], [], []

    for _, row in chain.iterrows():
        K = float(row["strike"])
        price = _mid_price(row)
        try:
            if price is None:
                raise ValueError("no usable quote")
            iv = implied_volatility(price, S, K, T, r, option_type, q)
            g = greeks(S, K, T, r, iv, option_type, q)
        except (ValueError, NoArbitrageViolation):
            ivs.append(float("nan"))
            deltas.append(float("nan"))
            gammas.append(float("nan"))
            vegas.append(float("nan"))
            thetas.append(float("nan"))
            rhos.append(float("nan"))
            continue
        ivs.append(iv)
        deltas.append(g.delta)
        gammas.append(g.gamma)
        vegas.append(g.vega)
        thetas.append(g.theta)
        rhos.append(g.rho)

    out["my_iv"] = ivs
    out["delta"] = deltas
    out["gamma"] = gammas
    out["vega"] = vegas
    out["theta"] = thetas
    out["rho"] = rhos
    return out
