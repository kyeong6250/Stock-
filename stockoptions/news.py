"""Recent news headlines -- an *informational* panel only.

Deliberately not wired into backtest.py's directional signal: turning
headlines into a sentiment feature has a mixed, easy-to-overfit track
record in the research, and validating it properly (walk-forward, no
lookahead) is its own project. This module exists so a human looking at
the dashboard can see what's being said about a ticker or a market-moving
figure, not so the model can "read the news" -- see the README's News
section for the reasoning.

Two sources, both used read-only against public/free endpoints (no ToS
risk, unlike social.py):

- yfinance's own news feed (`get_ticker_news`) -- no API key, always
  available, ticker-scoped. Same dependency data.py already uses.
- Finnhub's free-tier news endpoints (`get_market_news`, `search_mentions`)
  -- broader, ticker-agnostic market news, needs a free key from
  finnhub.io (see .env.example). Everything else in this module works
  with zero Finnhub key.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
import yfinance as yf
from dotenv import load_dotenv


class NewsFetchError(RuntimeError):
    """Raised when a news source can't be reached or isn't configured."""


@dataclass
class NewsItem:
    title: str
    summary: str
    publisher: str
    url: str
    published_at: datetime | None
    source: str  # "yfinance" | "finnhub"


def get_ticker_news(ticker: str, limit: int = 10) -> list[NewsItem]:
    """Recent headlines for one ticker, straight from Yahoo Finance via
    yfinance -- no API key needed. yfinance nests most fields under a
    "content" key as of the version this was built against (verified
    live, 2026-08-20); this reads defensively in case that key is ever
    missing rather than raising on one malformed entry."""
    raw = yf.Ticker(ticker).news or []
    items = []
    for entry in raw[:limit]:
        content = entry.get("content") or {}
        items.append(
            NewsItem(
                title=content.get("title", ""),
                summary=content.get("summary", ""),
                publisher=(content.get("provider") or {}).get("displayName", "Yahoo Finance"),
                url=(content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get("url", ""),
                published_at=_parse_iso8601(content.get("pubDate")),
                source="yfinance",
            )
        )
    return items


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_finnhub_key() -> str | None:
    load_dotenv()
    return os.environ.get("FINNHUB_API_KEY")


def get_market_news(category: str = "general", limit: int = 20) -> list[NewsItem]:
    """Broad market news (not ticker-specific) via Finnhub's free tier.
    `category` is one of Finnhub's own values: general/forex/crypto/merger.
    Raises NewsFetchError if FINNHUB_API_KEY isn't set -- get a free key
    at finnhub.io and put it in .env (see .env.example)."""
    key = _load_finnhub_key()
    if not key:
        raise NewsFetchError("FINNHUB_API_KEY isn't set (see .env.example) -- get a free key at finnhub.io")

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/news",
            params={"category": category, "token": key},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as exc:
        raise NewsFetchError(f"Finnhub request failed: {exc}") from exc
    if not isinstance(raw, list):
        raise NewsFetchError(f"unexpected Finnhub response shape: {raw!r}")

    items = []
    for entry in raw[:limit]:
        published_at = datetime.fromtimestamp(entry["datetime"], tz=timezone.utc) if entry.get("datetime") else None
        items.append(
            NewsItem(
                title=entry.get("headline", ""),
                summary=entry.get("summary", ""),
                publisher=entry.get("source", ""),
                url=entry.get("url", ""),
                published_at=published_at,
                source="finnhub",
            )
        )
    return items


def search_mentions(name: str, category: str = "general", scan_limit: int = 100, limit: int = 10) -> list[NewsItem]:
    """Headlines/summaries mentioning `name` (e.g. "Trump", "Musk") among
    the most recent `scan_limit` market-news items. This is a client-side
    substring filter over Finnhub's general feed, not a real search index
    -- Finnhub's free tier doesn't offer full-text search on this
    endpoint. Good for "did anything about this person hit the news
    recently," not a guarantee of completeness or of catching everything
    ever said about them."""
    needle = name.lower()
    candidates = get_market_news(category=category, limit=scan_limit)
    matches = [item for item in candidates if needle in item.title.lower() or needle in item.summary.lower()]
    return matches[:limit]
