import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from stockoptions.analysis import TickerOverview
from stockoptions.data import TickerNotFoundError
from stockoptions.news import NewsItem
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
