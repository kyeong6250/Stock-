"""Black-Scholes-Merton option pricing, Greeks, and implied volatility.

Standard continuous-dividend-yield formulation. All rates/vols are annualized
decimals (0.05 = 5%), T is time to expiration in years, q is the continuous
dividend yield (0 if you don't want to model dividends).

Vega is returned per 1.0 (100 percentage points) change in volatility, the
mathematical convention -- divide by 100 to get the "per 1% IV move" figure
traders usually quote. Theta is returned per year -- divide by 365 for the
"per calendar day" figure traders usually quote.
"""

from dataclasses import dataclass
from math import exp, log, pi, sqrt

from scipy.optimize import brentq
from scipy.stats import norm


class NoArbitrageViolation(ValueError):
    """Raised when a target price is outside the no-arbitrage bounds for the
    given inputs, so no implied volatility exists that could produce it."""


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must all be positive")
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float  # per 1.0 (100pp) change in sigma
    theta: float  # per year
    rho: float


def greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> Greeks:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    disc_q = exp(-q * T)
    disc_r = exp(-r * T)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt(T))
    vega = S * disc_q * pdf_d1 * sqrt(T)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt(T))
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1)
        theta = (
            -S * disc_q * pdf_d1 * sigma / (2 * sqrt(T))
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
        rho = -K * T * disc_r * norm.cdf(-d2)

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def _no_arbitrage_bounds(S: float, K: float, T: float, r: float, option_type: str, q: float = 0.0) -> tuple[float, float]:
    disc_r = exp(-r * T)
    disc_q = exp(-q * T)
    if option_type == "call":
        lower = max(0.0, S * disc_q - K * disc_r)
        upper = S * disc_q
    else:
        lower = max(0.0, K * disc_r - S * disc_q)
        upper = K * disc_r
    return lower, upper


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
    *,
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Solve for the volatility that reproduces `price` under Black-Scholes.

    Raises NoArbitrageViolation if `price` sits outside the model's
    no-arbitrage bounds for these inputs (e.g. a quote below intrinsic
    value) -- no sigma, however extreme, could produce it."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    lower_bound, upper_bound = _no_arbitrage_bounds(S, K, T, r, option_type, q)
    if not (lower_bound < price < upper_bound):
        raise NoArbitrageViolation(
            f"price {price} is outside no-arbitrage bounds ({lower_bound:.4f}, {upper_bound:.4f}) "
            f"for {option_type} S={S} K={K} T={T} r={r} q={q}"
        )

    pricer = call_price if option_type == "call" else put_price

    def objective(sigma: float) -> float:
        return pricer(S, K, T, r, sigma, q) - price

    return brentq(objective, lo, hi, xtol=1e-8)
