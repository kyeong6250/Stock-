"""Cox-Ross-Rubinstein binomial tree pricing for American-style options.

Why this exists alongside blackscholes.py: real US equity options are
American-style (exercisable any time before expiration), not European
(exercisable only at expiration, which is what Black-Scholes actually
prices). Using European Black-Scholes to solve for the implied volatility
of an American-style market quote is a real, well-documented source of
error -- it's the standard textbook justification for the binomial model,
confirmed while researching this: "the binomial model offers ... the
ability to price American options ... [handling] early exercise", and
"with a few hundred steps the implementation yields four-decimal accuracy
for vanilla American puts and calls on dividend-paying stocks" (see the
README's Accuracy upgrades section for sources).

blackscholes.py isn't obsolete here -- it's still the right tool for
strategies.py (payoff *at expiration* is just intrinsic value, exercise
style is irrelevant there) and for anyone who wants the faster, simpler
European approximation. This module is specifically for pricing/IV/Greeks
*before* expiration, where American-vs-European actually matters.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

DEFAULT_STEPS = 150


class NoArbitrageViolation(ValueError):
    """Raised when a target price can't be matched by any volatility in
    the search range -- mirrors blackscholes.NoArbitrageViolation."""


def _validate(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must all be positive")


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float  # per year
    vega: float  # per 1.0 (100pp) change in sigma
    rho: float  # per 1.0 (100pp) change in r


def price(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, steps: int = DEFAULT_STEPS) -> float:
    """American option price via a CRR binomial tree with early exercise
    checked at every node."""
    _validate(S, K, T, sigma)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0 < p < 1):
        raise ValueError(
            f"risk-neutral probability {p:.4f} is outside (0,1) -- check r/q/sigma/steps "
            "(a very large r-q relative to sigma*sqrt(dt) can cause this)"
        )

    j = np.arange(steps + 1)
    ST = S * (u**j) * (d ** (steps - j))
    values = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        jj = np.arange(i + 1)
        St = S * (u**jj) * (d ** (i - jj))
        intrinsic = np.maximum(St - K, 0.0) if option_type == "call" else np.maximum(K - St, 0.0)
        values = np.maximum(values, intrinsic)

    return float(values[0])


def price_and_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0, steps: int = DEFAULT_STEPS
) -> tuple[float, Greeks]:
    """Same tree as price(), but keeps the step-1 and step-2 node values
    around to extract delta/gamma/theta directly from the tree (the
    standard technique -- see e.g. Hull's Options, Futures, and Other
    Derivatives), which is both cheaper and more consistent than bumping
    S and rebuilding the whole tree. Vega and rho still use bump-and-
    reprice (central difference), since there's no equally clean way to
    read them off a single tree."""
    _validate(S, K, T, sigma)
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if steps < 2:
        raise ValueError("steps must be >= 2 to extract Greeks from the tree")

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0 < p < 1):
        raise ValueError(f"risk-neutral probability {p:.4f} is outside (0,1) -- check r/q/sigma/steps")

    j = np.arange(steps + 1)
    ST = S * (u**j) * (d ** (steps - j))
    values = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)

    saved: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        jj = np.arange(i + 1)
        St = S * (u**jj) * (d ** (i - jj))
        intrinsic = np.maximum(St - K, 0.0) if option_type == "call" else np.maximum(K - St, 0.0)
        values = np.maximum(values, intrinsic)
        if i <= 2:
            saved[i] = (values.copy(), St.copy())

    option_price = float(saved[0][0][0])
    V1, S1 = saved[1]
    V2, S2 = saved[2]

    delta = (V1[1] - V1[0]) / (S1[1] - S1[0])
    gamma = ((V2[2] - V2[1]) / (S2[2] - S2[1]) - (V2[1] - V2[0]) / (S2[1] - S2[0])) / (0.5 * (S2[2] - S2[0]))
    theta = (V2[1] - option_price) / (2 * dt)

    eps_sigma = max(1e-4, sigma * 0.01)
    vega = (price(S, K, T, r, sigma + eps_sigma, option_type, q, steps) - price(S, K, T, r, sigma - eps_sigma, option_type, q, steps)) / (
        2 * eps_sigma
    )

    eps_r = 1e-4
    rho = (price(S, K, T, r + eps_r, sigma, option_type, q, steps) - price(S, K, T, r - eps_r, sigma, option_type, q, steps)) / (2 * eps_r)

    return option_price, Greeks(delta=float(delta), gamma=float(gamma), theta=float(theta), vega=float(vega), rho=float(rho))


def implied_volatility(
    target_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
    steps: int = 100,
    *,
    lo: float = 0.01,
    hi: float = 5.0,
) -> float:
    """Solve for the volatility that reproduces `target_price` under the
    American binomial model. Uses fewer steps than price_and_greeks by
    default (100 vs 150) since Brent's method calls the pricer dozens of
    times per solve and this typically runs across a whole option chain;
    still comfortably within the "few hundred steps" accuracy range from
    the research this module is based on.

    `lo` defaults to 1% annualized vol rather than something closer to 0:
    the CRR risk-neutral probability p = (e^{(r-q)dt} - d) / (u - d) can
    fall outside (0,1) -- a known numerical characteristic of the standard
    CRR parameterization -- when sigma*sqrt(dt) is small relative to
    (r-q)*dt, i.e. exactly at the very-low-volatility end of the search
    range. 1% is comfortably below any volatility a real option would
    ever actually imply, so this costs nothing in practice."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    lower_price = price(S, K, T, r, lo, option_type, q, steps)
    upper_price = price(S, K, T, r, hi, option_type, q, steps)
    if not (lower_price <= target_price <= upper_price):
        raise NoArbitrageViolation(
            f"price {target_price} is outside the achievable range ({lower_price:.4f}, {upper_price:.4f}) "
            f"for sigma in [{lo}, {hi}] with {option_type} S={S} K={K} T={T} r={r} q={q}"
        )

    def objective(sigma: float) -> float:
        return price(S, K, T, r, sigma, option_type, q, steps) - target_price

    return brentq(objective, lo, hi, xtol=1e-6)
