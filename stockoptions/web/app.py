"""FastAPI backend for the local dashboard (run via `stockoptions dashboard`).

Thin JSON wrappers around the same tested stockoptions modules the CLI
uses -- no business logic lives here, just request handling and error
translation into HTTP status codes. Every number here traces back to a
pure, unit-tested function elsewhere in the package.
"""

import math
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from stockoptions.analysis import screen_ticker
from stockoptions.backtest import backtest
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
from stockoptions.news import get_ticker_news
from stockoptions.rates import risk_free_rate
from stockoptions.social import SocialFetchError, get_truth_social_posts, get_x_posts
from stockoptions.strategies import (
    breakevens,
    default_scan_range,
    iron_condor,
    max_loss,
    max_profit,
    payoff_curve,
    strangle,
    vertical_spread,
)

app = FastAPI(title="stockoptions dashboard")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/overview/{ticker}")
def api_overview(ticker: str) -> dict:
    try:
        overview = screen_ticker(ticker.upper())
    except TickerNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {
        "ticker": overview.ticker,
        "price": overview.price,
        "atmIv": overview.atm_iv,
        "realizedVol30d": overview.realized_vol_30d,
        "ivHvRatio": overview.iv_hv_ratio,
        "read": overview.read,
        "riskFreeRate": overview.risk_free_rate,
        "dividendYield": overview.dividend_yield,
        "nearestExpiration": overview.nearest_expiration,
    }


@app.get("/api/history/{ticker}")
def api_history(ticker: str, period: str = "6mo") -> list[dict]:
    try:
        df = get_price_history(ticker.upper(), period=period)
    except TickerNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return [{"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"])} for idx, row in df.iterrows()]


@app.get("/api/expirations/{ticker}")
def api_expirations(ticker: str) -> list[str]:
    try:
        return get_option_expirations(ticker.upper())
    except TickerNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/chain/{ticker}")
def api_chain(ticker: str, expiration: str = Query(...), type: str = Query("call")) -> dict:
    ticker = ticker.upper()
    if type not in ("call", "put"):
        raise HTTPException(400, "type must be 'call' or 'put'")
    try:
        S = get_current_price(ticker)
        calls, puts = get_option_chain(ticker, expiration)
        chain = calls if type == "call" else puts
        T = years_to_expiration(expiration)
        r = risk_free_rate(T)
        q = get_dividend_yield(ticker)
        enriched = enrich_chain_with_iv_and_greeks(chain, S, expiration, type, r, q)
    except (TickerNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    rows = []
    for _, row in enriched.iterrows():
        if math.isnan(row["my_iv"]):
            continue
        rows.append(
            {
                "strike": float(row["strike"]),
                "lastPrice": float(row["lastPrice"]),
                "iv": float(row["my_iv"]),
                "delta": float(row["delta"]),
                "gamma": float(row["gamma"]),
                "vega": float(row["vega"]),
                "theta": float(row["theta"]),
            }
        )
    return {"spot": S, "riskFreeRate": r, "dividendYield": q, "rows": rows}


@app.get("/api/backtest/{ticker}")
def api_backtest(ticker: str, period: str = "2y", horizon: int = 5) -> dict:
    try:
        history = get_price_history(ticker.upper(), period=period)
        result = backtest(history, horizon_days=horizon)
    except (TickerNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "accuracy": result.accuracy,
        "majorityBaseline": result.majority_baseline_accuracy,
        "beatsBaseline": result.beats_baseline,
        "nTrain": result.n_train_samples,
        "nTest": result.n_test_samples,
        "strategyReturn": result.strategy_total_return,
        "buyHoldReturn": result.buy_and_hold_total_return,
        "sharpe": result.strategy_sharpe,
        "maxDrawdown": result.strategy_max_drawdown,
        "dates": result.dates,
        "strategyCurve": result.strategy_equity_curve,
        "buyHoldCurve": result.buy_and_hold_equity_curve,
    }


@app.post("/api/strategy")
def api_strategy(payload: dict = Body(...)) -> dict:
    kind = payload.get("kind")
    try:
        if kind == "vertical":
            strat = vertical_spread(
                payload["optionType"], payload["longStrike"], payload["shortStrike"], payload["longPremium"], payload["shortPremium"]
            )
        elif kind == "strangle":
            strat = strangle(
                payload["callStrike"], payload["putStrike"], payload["callPremium"], payload["putPremium"], short=not payload.get("long", False)
            )
        elif kind == "iron-condor":
            strat = iron_condor(
                payload["putLongStrike"],
                payload["putShortStrike"],
                payload["callShortStrike"],
                payload["callLongStrike"],
                payload["putLongPremium"],
                payload["putShortPremium"],
                payload["callShortPremium"],
                payload["callLongPremium"],
            )
        else:
            raise HTTPException(400, f"unknown strategy kind {kind!r}, expected vertical/strangle/iron-condor")
    except KeyError as exc:
        raise HTTPException(400, f"missing required field: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    profit = max_profit(strat)
    loss = max_loss(strat)
    bes = breakevens(strat)
    lo, hi = default_scan_range(strat)
    xs, ys = payoff_curve(strat, lo, hi, num=200)

    return {
        "name": strat.name,
        "netPremium": strat.net_premium(),
        "maxProfit": None if profit == float("inf") else profit,
        "maxProfitUnlimited": profit == float("inf"),
        "maxLoss": None if loss == float("-inf") else loss,
        "maxLossUnlimited": loss == float("-inf"),
        "breakevens": bes,
        "curve": [{"s": float(x), "pnl": float(y)} for x, y in zip(xs, ys)],
    }


@app.get("/api/news/{ticker}")
def api_news(ticker: str, limit: int = 8) -> list[dict]:
    items = get_ticker_news(ticker.upper(), limit=limit)
    return [
        {
            "title": item.title,
            "summary": item.summary,
            "publisher": item.publisher,
            "url": item.url,
            "publishedAt": item.published_at.isoformat() if item.published_at else None,
            "source": item.source,
        }
        for item in items
    ]


# Well-known figures whose public statements have a track record of moving
# specific tickers/sectors -- shown on the Influencer watch panel. This is
# a fixed, curated list (not user-configurable input) specifically so
# nothing here becomes a free-text field that gets passed to social.py's
# unofficial scrapers -- see sanitizeTicker()/escapeHtml() in app.js for
# why unvalidated free text into a scraper or the DOM is exactly the kind
# of thing this project has already had to defend against once.
INFLUENCER_WATCHLIST = [
    {"platform": "truth", "handle": "realDonaldTrump", "label": "Donald Trump"},
    {"platform": "x", "handle": "elonmusk", "label": "Elon Musk"},
]


@app.get("/api/influencers")
def api_influencers(limit: int = 5) -> list[dict]:
    """Best-effort pull per figure -- one source failing (most likely: X,
    see social.py's docstring) doesn't take down the others."""
    results = []
    for entry in INFLUENCER_WATCHLIST:
        try:
            fetch = get_truth_social_posts if entry["platform"] == "truth" else get_x_posts
            posts = fetch(entry["handle"], limit=limit)
            results.append(
                {
                    "platform": entry["platform"],
                    "handle": entry["handle"],
                    "label": entry["label"],
                    "available": True,
                    "posts": [
                        {"text": p.text, "url": p.url, "postedAt": p.posted_at.isoformat() if p.posted_at else None}
                        for p in posts
                    ],
                }
            )
        except SocialFetchError as exc:
            results.append(
                {"platform": entry["platform"], "handle": entry["handle"], "label": entry["label"], "available": False, "error": str(exc), "posts": []}
            )
    return results


# Mounted last so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
