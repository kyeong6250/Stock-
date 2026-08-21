"""Scan several tickers at once -- a "which of these is even worth
looking at closer" pass, rather than typing one ticker at a time into
every other panel.

Deliberately built on the *cheap* per-ticker building blocks
(analysis.screen_ticker, recommend.predict_live_direction, backtest.backtest)
rather than recommend.py's full recommend_trade() pipeline: real
delta-targeted chain enrichment across every strike plus the Kelly
replay's per-row binomial pricing loop only make sense once you've
already picked one ticker to commit real analysis time to, not across a
whole watchlist.

DEFAULT_WATCHLIST is a fixed, curated seed list of large, liquid names
across sectors -- not a "trending" or "hot stocks" claim pulled from some
external service. No free, trustworthy "what's trending right now" API
exists (this project already learned that lesson the hard way with
Truth Social/X in social.py: don't assume a "free" data source works
without checking). What actually earns a ticker a "hot" ranking here is
real, computed data pulled at scan time -- today's trading volume
relative to its own 20-day average -- not an external assertion.
"""

from dataclasses import dataclass

from stockoptions.analysis import screen_ticker
from stockoptions.backtest import backtest
from stockoptions.data import TickerNotFoundError, get_price_history
from stockoptions.rates import get_yield_curve
from stockoptions.recommend import RecommendationError, predict_live_direction

# Large, liquid names across sectors, chosen so a scan always has a real,
# thickly-traded options chain to work with -- not a claim about which of
# these is currently a good trade. See this module's docstring.
DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AVGO", "JPM", "XOM", "UNH", "V", "WMT", "KO", "DIS",
]


@dataclass
class TickerScanResult:
    ticker: str
    price: float | None = None
    volume_ratio: float | None = None  # today's volume / trailing-20-day average - 1; None if unavailable
    iv_hv_ratio: float | None = None
    read: str | None = None  # "rich" | "cheap" | "in line"
    direction: str | None = None  # "up" | "down"
    live_probability: float | None = None
    backtest_accuracy: float | None = None
    backtest_baseline: float | None = None
    beats_baseline: bool | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _today_volume_ratio(history) -> float | None:
    """Today's volume vs. its own trailing 20-day average, as a fraction
    (0.5 = 50% above average). This IS the honest "hot" signal a scan
    ranks by -- real, computed from the same history already fetched for
    the directional model, not sourced from anywhere claiming to know
    what's "trending." None if there isn't enough history to compute a
    trailing average yet."""
    volume = history["Volume"]
    if len(volume) < 21:
        return None
    avg20 = volume.iloc[-21:-1].mean()
    if avg20 <= 0:
        return None
    return float(volume.iloc[-1] / avg20 - 1)


def scan_tickers(tickers: list[str], horizon_days: int = 5, model_type: str = "logistic") -> list[TickerScanResult]:
    """Runs screen_ticker + today's directional call + a backtest for
    each ticker, isolating failures per-ticker (one illiquid/delisted
    symbol shouldn't kill the whole scan -- same pattern as
    /api/influencers) rather than raising and losing every other result.
    Fetches the Treasury yield curve once and reuses it across tickers,
    the same optimization screen_ticker's own docstring already
    recommends for multi-ticker use."""
    curve = get_yield_curve()
    results = []
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        try:
            overview = screen_ticker(ticker, yield_curve=curve)
            history = get_price_history(ticker, period="2y")
            volume_ratio = _today_volume_ratio(history)
            direction, probability = predict_live_direction(history, horizon_days, model_type=model_type)
            bt = backtest(history, horizon_days=horizon_days, model_type=model_type)
            results.append(
                TickerScanResult(
                    ticker=ticker,
                    price=overview.price,
                    volume_ratio=volume_ratio,
                    iv_hv_ratio=overview.iv_hv_ratio,
                    read=overview.read,
                    direction=direction,
                    live_probability=probability,
                    backtest_accuracy=bt.accuracy,
                    backtest_baseline=bt.majority_baseline_accuracy,
                    beats_baseline=bt.beats_baseline,
                )
            )
        except (TickerNotFoundError, ValueError, RecommendationError) as exc:
            results.append(TickerScanResult(ticker=ticker, error=str(exc)))
    return results
