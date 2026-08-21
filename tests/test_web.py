import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from stockoptions.analysis import TickerOverview
from stockoptions.data import TickerNotFoundError
from stockoptions.backtest import BacktestResult
from stockoptions.news import NewsItem
from stockoptions.recommend import ConePoint, KellyEdge, RecommendationError, SelectedContract, TradeRecommendation
from stockoptions.social import SocialFetchError, SocialPost
from stockoptions.web.app import app

LIVE = os.environ.get("STOCKOPTIONS_LIVE_TESTS") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE, reason="set STOCKOPTIONS_LIVE_TESTS=1 to run tests that hit the real yfinance API")

client = TestClient(app)


def test_index_serves_the_dashboard_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "stockoptions dashboard" in res.text


def test_strategy_endpoint_iron_condor_matches_hand_verified_numbers():
    # Same numbers hand-verified in test_strategies.py and test_cli.py --
    # confirms the API wraps the exact same engine, not a reimplementation.
    payload = {
        "kind": "iron-condor",
        "putLongStrike": 90, "putShortStrike": 95, "callShortStrike": 105, "callLongStrike": 110,
        "putLongPremium": 1, "putShortPremium": 2, "callShortPremium": 2, "callLongPremium": 1,
    }
    res = client.post("/api/strategy", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["maxProfit"] == pytest.approx(2.0)
    assert body["maxLoss"] == pytest.approx(-3.0)
    assert sorted(body["breakevens"]) == pytest.approx([93.0, 107.0], abs=0.01)
    assert body["maxProfitUnlimited"] is False
    assert len(body["curve"]) > 0


def test_strategy_endpoint_reports_unbounded_loss_as_null_with_a_flag():
    # A short strangle has a naked short call in it -- unbounded loss on
    # the upside, same as test_strategies.py's equivalent case.
    payload = {"kind": "strangle", "callStrike": 110, "putStrike": 90, "callPremium": 2, "putPremium": 2}
    res = client.post("/api/strategy", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["maxLossUnlimited"] is True
    assert body["maxLoss"] is None


def test_strategy_endpoint_rejects_unknown_kind():
    res = client.post("/api/strategy", json={"kind": "banana"})
    assert res.status_code == 400


def test_strategy_endpoint_rejects_missing_field_with_clear_message():
    res = client.post("/api/strategy", json={"kind": "vertical", "optionType": "call"})
    assert res.status_code == 400
    assert "missing required field" in res.json()["detail"]


def test_overview_endpoint_translates_ticker_not_found_to_404():
    with patch("stockoptions.web.app.screen_ticker", side_effect=TickerNotFoundError("no such ticker 'ZZZZZZ'")):
        res = client.get("/api/overview/ZZZZZZ")
    assert res.status_code == 404


def test_overview_endpoint_shape_with_a_mocked_screen_ticker():
    fake = TickerOverview(
        ticker="AAPL", price=316.83, atm_iv=0.25, realized_vol_30d=0.32, iv_hv_ratio=0.78,
        read="cheap", risk_free_rate=0.037, dividend_yield=0.0034, nearest_expiration="2026-09-18",
    )
    with patch("stockoptions.web.app.screen_ticker", return_value=fake):
        res = client.get("/api/overview/AAPL")
    assert res.status_code == 200
    body = res.json()
    assert body["ticker"] == "AAPL"
    assert body["read"] == "cheap"
    assert body["riskFreeRate"] == pytest.approx(0.037)


@skip_unless_live
def test_overview_endpoint_live():
    res = client.get("/api/overview/AAPL")
    assert res.status_code == 200
    body = res.json()
    assert body["price"] > 0
    assert body["read"] in ("rich", "cheap", "in line")


@skip_unless_live
def test_history_endpoint_live():
    res = client.get("/api/history/AAPL?period=1mo")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) > 0
    assert "date" in rows[0] and "close" in rows[0]


def test_news_endpoint_shape_with_mocked_ticker_news():
    fake_items = [
        NewsItem(title="Widget Co beats earnings", summary="...", publisher="Example Wire", url="https://example.com/a", published_at=None, source="yfinance")
    ]
    with patch("stockoptions.web.app.get_ticker_news", return_value=fake_items) as mock_fn:
        res = client.get("/api/news/AAPL?limit=3")
    assert res.status_code == 200
    mock_fn.assert_called_once_with("AAPL", limit=3)
    body = res.json()
    assert body[0]["title"] == "Widget Co beats earnings"
    assert body[0]["publisher"] == "Example Wire"


def test_influencers_endpoint_isolates_a_failing_source_from_a_working_one():
    fake_posts = [SocialPost(platform="truth_social", author="realDonaldTrump", text="hello", url="https://x", posted_at=None)]
    with (
        patch("stockoptions.web.app.get_truth_social_posts", return_value=fake_posts),
        patch("stockoptions.web.app.get_x_posts", side_effect=SocialFetchError("no working instance")),
    ):
        res = client.get("/api/influencers")
    assert res.status_code == 200
    body = res.json()
    truth_entry = next(e for e in body if e["platform"] == "truth")
    x_entry = next(e for e in body if e["platform"] == "x")
    assert truth_entry["available"] is True
    assert truth_entry["posts"][0]["text"] == "hello"
    assert x_entry["available"] is False
    assert "no working instance" in x_entry["error"]
    assert x_entry["posts"] == []


def _fake_recommendation():
    bt = BacktestResult(
        accuracy=0.45,
        majority_baseline_accuracy=0.55,
        n_train_samples=100,
        n_test_samples=40,
        strategy_total_return=0.1,
        buy_and_hold_total_return=0.05,
        strategy_sharpe=1.2,
        strategy_max_drawdown=-0.05,
        dates=["2026-01-01"],
        strategy_equity_curve=[1.1],
        buy_and_hold_equity_curve=[1.05],
    )
    contract = SelectedContract(
        expiration="2026-09-25", dte=36, option_type="call", strike=325.0, premium=4.7, iv=0.245, delta=0.316, T=0.0986, r=0.037
    )
    edge = KellyEdge(sample_size=49, win_rate=0.33, avg_win=13.5, avg_loss=3.8, kelly_fraction=0.14, insufficient_sample=False)
    cone = [ConePoint(day=1, upper_1sd=105.0, lower_1sd=95.0, upper_2sd=110.0, lower_2sd=90.0)]
    return TradeRecommendation(
        ticker="AAPL",
        price=100.0,
        direction="up",
        live_probability=0.576,
        backtest=bt,
        contract=contract,
        edge=edge,
        kelly_multiplier=0.5,
        max_risk_pct=0.02,
        recommended_risk_fraction=0.02,
        recommended_dollar_risk=2000.0,
        recommended_contracts=4,
        actual_dollar_risk=1880.0,
        cone=cone,
        warnings=["did not beat baseline"],
    )


def test_predict_endpoint_shape_with_a_mocked_recommendation():
    with patch("stockoptions.web.app.recommend_trade", return_value=_fake_recommendation()) as mock_fn:
        res = client.get("/api/predict/AAPL?account_size=50000&risk_pct=0.03&horizon=20&delta=0.4")
    assert res.status_code == 200
    mock_fn.assert_called_once_with("AAPL", account_size=50000.0, max_risk_pct=0.03, horizon_days=20, target_delta=0.4)
    body = res.json()
    assert body["ticker"] == "AAPL"
    assert body["direction"] == "up"
    assert body["contract"]["strike"] == 325.0
    assert body["sizing"]["contracts"] == 4
    assert len(body["cone"]) == 1
    assert body["cone"][0]["upper1"] == 105.0
    assert body["warnings"] == ["did not beat baseline"]


def test_predict_endpoint_translates_recommendation_error_to_400():
    with patch("stockoptions.web.app.recommend_trade", side_effect=RecommendationError("no usable contracts")):
        res = client.get("/api/predict/ZZZZZZ")
    assert res.status_code == 400
    assert "no usable contracts" in res.json()["detail"]


@skip_unless_live
def test_predict_endpoint_live():
    res = client.get("/api/predict/AAPL")
    assert res.status_code == 200
    body = res.json()
    assert body["direction"] in ("up", "down")
    assert body["contract"]["strike"] > 0
    assert len(body["cone"]) > 0


@skip_unless_live
def test_news_endpoint_live():
    res = client.get("/api/news/AAPL?limit=3")
    assert res.status_code == 200
    body = res.json()
    assert len(body) > 0
    assert body[0]["title"]


@skip_unless_live
def test_backtest_endpoint_live():
    res = client.get("/api/backtest/AAPL?period=2y&horizon=5")
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["accuracy"] <= 1.0
    assert len(body["dates"]) == len(body["strategyCurve"]) == len(body["buyHoldCurve"])
