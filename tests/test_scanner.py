from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from stockoptions.analysis import TickerOverview
from stockoptions.backtest import BacktestResult
from stockoptions.data import TickerNotFoundError
from stockoptions.recommend import RecommendationError
from stockoptions.scanner import DEFAULT_WATCHLIST, _today_volume_ratio, scan_tickers


def _fake_overview(ticker, price=100.0, iv_hv_ratio=1.0, read="in line"):
    return TickerOverview(
        ticker=ticker, price=price, atm_iv=0.3, realized_vol_30d=0.3, iv_hv_ratio=iv_hv_ratio,
        read=read, risk_free_rate=0.04, dividend_yield=0.0, nearest_expiration="2026-12-31",
    )


def _fake_backtest_result(accuracy=0.5, baseline=0.5):
    return BacktestResult(
        accuracy=accuracy, majority_baseline_accuracy=baseline, n_train_samples=100, n_test_samples=40,
        strategy_total_return=0.1, buy_and_hold_total_return=0.05, strategy_sharpe=1.0, strategy_max_drawdown=-0.05,
        dates=["2026-01-01"], strategy_equity_curve=[1.1], buy_and_hold_equity_curve=[1.05], feature_importance={},
    )


def _fake_history(n=40):
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    volume = np.full(n, 1_000_000.0)
    volume[-1] = 2_000_000.0  # today's volume is 2x a flat trailing average
    return pd.DataFrame({"Close": np.full(n, 100.0), "Volume": volume}, index=dates)


def test_scan_tickers_returns_one_populated_result_per_ticker():
    with (
        patch("stockoptions.scanner.get_yield_curve", return_value={1.0: 0.04}),
        patch("stockoptions.scanner.screen_ticker", side_effect=lambda t, yield_curve=None: _fake_overview(t)),
        patch("stockoptions.scanner.get_price_history", return_value=_fake_history()),
        patch("stockoptions.scanner.predict_live_direction", return_value=("up", 0.6)),
        patch("stockoptions.scanner.backtest", return_value=_fake_backtest_result(accuracy=0.55, baseline=0.5)),
    ):
        results = scan_tickers(["AAPL", "MSFT"])

    assert len(results) == 2
    assert all(r.ok for r in results)
    assert {r.ticker for r in results} == {"AAPL", "MSFT"}
    aapl = next(r for r in results if r.ticker == "AAPL")
    assert aapl.price == 100.0
    assert aapl.direction == "up"
    assert aapl.live_probability == 0.6
    assert aapl.backtest_accuracy == 0.55
    assert aapl.beats_baseline is True
    assert aapl.volume_ratio == pytest.approx(1.0)  # 2,000,000 / 1,000,000 average - 1


def test_scan_tickers_isolates_a_failing_ticker_from_the_others():
    def fake_screen(ticker, yield_curve=None):
        if ticker == "BADTICKER":
            raise TickerNotFoundError("no data for BADTICKER")
        return _fake_overview(ticker)

    with (
        patch("stockoptions.scanner.get_yield_curve", return_value={1.0: 0.04}),
        patch("stockoptions.scanner.screen_ticker", side_effect=fake_screen),
        patch("stockoptions.scanner.get_price_history", return_value=_fake_history()),
        patch("stockoptions.scanner.predict_live_direction", return_value=("up", 0.6)),
        patch("stockoptions.scanner.backtest", return_value=_fake_backtest_result()),
    ):
        results = scan_tickers(["AAPL", "BADTICKER", "MSFT"])

    assert len(results) == 3
    good = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    assert {r.ticker for r in good} == {"AAPL", "MSFT"}
    assert bad[0].ticker == "BADTICKER"
    assert "no data for BADTICKER" in bad[0].error


def test_scan_tickers_also_isolates_a_recommendation_error_from_predict_live_direction():
    with (
        patch("stockoptions.scanner.get_yield_curve", return_value={1.0: 0.04}),
        patch("stockoptions.scanner.screen_ticker", side_effect=lambda t, yield_curve=None: _fake_overview(t)),
        patch("stockoptions.scanner.get_price_history", return_value=_fake_history()),
        patch("stockoptions.scanner.predict_live_direction", side_effect=RecommendationError("not enough history")),
        patch("stockoptions.scanner.backtest", return_value=_fake_backtest_result()),
    ):
        results = scan_tickers(["AAPL"])

    assert len(results) == 1
    assert not results[0].ok
    assert "not enough history" in results[0].error


def test_scan_tickers_skips_blank_entries():
    with (
        patch("stockoptions.scanner.get_yield_curve", return_value={1.0: 0.04}),
        patch("stockoptions.scanner.screen_ticker", side_effect=lambda t, yield_curve=None: _fake_overview(t)),
        patch("stockoptions.scanner.get_price_history", return_value=_fake_history()),
        patch("stockoptions.scanner.predict_live_direction", return_value=("up", 0.6)),
        patch("stockoptions.scanner.backtest", return_value=_fake_backtest_result()),
    ):
        results = scan_tickers(["AAPL", "", "   "])

    assert len(results) == 1
    assert results[0].ticker == "AAPL"


def test_today_volume_ratio_matches_hand_computed_value():
    history = _fake_history(n=30)
    ratio = _today_volume_ratio(history)
    assert ratio == pytest.approx(1.0)  # today 2,000,000 vs trailing-20 average of 1,000,000


def test_today_volume_ratio_is_none_with_insufficient_history():
    history = _fake_history(n=10)
    assert _today_volume_ratio(history) is None


def test_default_watchlist_is_a_clean_uppercase_list_with_no_duplicates():
    assert len(DEFAULT_WATCHLIST) > 0
    assert all(t == t.upper() for t in DEFAULT_WATCHLIST)
    assert len(DEFAULT_WATCHLIST) == len(set(DEFAULT_WATCHLIST))
