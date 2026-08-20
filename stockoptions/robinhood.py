"""Read-only Robinhood integration: pull your watchlist/positions so the
rest of the toolkit has tickers to screen, nothing more.

This module only ever CALLS read-style functions (login/logout and the
get_*/build_* functions on robin_stocks.robinhood.account) -- never
anything from robin_stocks.robinhood.orders, where every order-placing,
-modifying, and -cancelling function lives, and never the watchlist
*write* functions (post_symbols_to_watchlist, delete_symbols_from_watchlist)
that happen to live alongside the read ones in .account.

Note this safety boundary is behavioral, not structural: robin_stocks'
own robinhood/__init__.py unconditionally imports every submodule,
including orders, the moment Python loads the package at all -- so
"this module never imports orders" turned out to be false the first time
it was actually tested (see test_robinhood.py), regardless of which
specific names get imported here. The real, honest guarantee is narrower:
none of robin_stocks' order-placing or watchlist-modifying function names
appear anywhere in this file's source, so it has no way to call them --
verified by tests/test_robinhood.py scanning this file's own source text.

Real risk, not eliminated by being read-only: Robinhood has no official
public API. robin_stocks reverse-engineers their private app API, which
violates Robinhood's Terms of Service, and automated access -- even
read-only -- can still trip their bot detection and get an account
flagged or suspended. That's a real risk you're taking on by using this
module at all, not something "read-only" makes go away.

Credentials load from environment variables (see .env.example) via
python-dotenv. Never hardcoded, never logged, never committed. The
session/device-token cache Robinhood's MFA flow produces is pointed at
~/.cache/stockoptions (outside any git repo) rather than trusting
robin_stocks' own default location.
"""

import os

from dotenv import load_dotenv
from robin_stocks.robinhood.account import build_holdings, get_all_watchlists, get_open_stock_positions, get_watchlist_by_name
from robin_stocks.robinhood.authentication import login, logout

from stockoptions.data import CACHE_DIR


class RobinhoodAuthError(RuntimeError):
    """Raised when login fails or required credentials aren't set."""


def _load_credentials() -> tuple[str, str, str | None]:
    load_dotenv()
    username = os.environ.get("ROBINHOOD_USERNAME")
    password = os.environ.get("ROBINHOOD_PASSWORD")
    mfa_code = os.environ.get("ROBINHOOD_MFA_CODE")  # optional; robin_stocks will prompt interactively if omitted
    if not username or not password:
        raise RobinhoodAuthError(
            "ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD must be set (see .env.example). "
            "Never hardcode credentials in source."
        )
    return username, password, mfa_code


def connect() -> None:
    """Log in, caching the session under ~/.cache/stockoptions so you're
    not re-entering MFA every run. Call disconnect() when done."""
    username, password, mfa_code = _load_credentials()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        login(username=username, password=password, mfa_code=mfa_code, pickle_path=str(CACHE_DIR))
    except Exception as exc:  # robin_stocks doesn't expose a narrow exception type
        raise RobinhoodAuthError(f"Robinhood login failed: {exc}") from exc


def disconnect() -> None:
    logout()


def get_watchlist_tickers(name: str = "My First List") -> list[str]:
    """Ticker symbols on one of your watchlists (default: Robinhood's
    default watchlist name)."""
    result = get_watchlist_by_name(name=name)
    if not result:
        raise ValueError(f"no watchlist named {name!r} found (or it's empty)")
    tickers = []
    for item in result:
        instrument_data = item.get("symbol") or (item.get("instrument_data") or {}).get("symbol")
        if instrument_data:
            tickers.append(instrument_data)
    return tickers


def get_watchlist_names() -> list[str]:
    watchlists = get_all_watchlists()
    return [w["display_name"] for w in watchlists.get("results", []) if "display_name" in w]


def get_position_tickers() -> list[str]:
    """Ticker symbols for your current open stock positions."""
    holdings = build_holdings()
    return list(holdings.keys())
