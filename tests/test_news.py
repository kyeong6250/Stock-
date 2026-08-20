import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from stockoptions.news import NewsFetchError, get_market_news, get_ticker_news, search_mentions

LIVE = os.environ.get("STOCKOPTIONS_LIVE_TESTS") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE, reason="set STOCKOPTIONS_LIVE_TESTS=1 to run tests that hit real news APIs")

_FAKE_YF_ARTICLE = {
    "content": {
        "title": "Widget Co beats earnings estimates",
        "summary": "Widget Co reported earnings above analyst expectations.",
        "pubDate": "2026-08-19T14:40:36Z",
        "provider": {"displayName": "Example Wire"},
        "canonicalUrl": {"url": "https://example.com/widget-earnings"},
    }
}


def test_get_ticker_news_parses_the_current_yfinance_schema():
    with patch("stockoptions.news.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = [_FAKE_YF_ARTICLE]
        items = get_ticker_news("WIDG", limit=5)

    assert len(items) == 1
    item = items[0]
    assert item.title == "Widget Co beats earnings estimates"
    assert item.publisher == "Example Wire"
    assert item.url == "https://example.com/widget-earnings"
    assert item.published_at == datetime(2026, 8, 19, 14, 40, 36, tzinfo=timezone.utc)
    assert item.source == "yfinance"


def test_get_ticker_news_tolerates_a_malformed_entry_missing_content():
    with patch("stockoptions.news.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = [{}]
        items = get_ticker_news("WIDG")
    assert items[0].title == ""
    assert items[0].published_at is None


def test_get_ticker_news_respects_limit():
    with patch("stockoptions.news.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.news = [_FAKE_YF_ARTICLE] * 20
        items = get_ticker_news("WIDG", limit=3)
    assert len(items) == 3


def test_get_market_news_raises_a_clear_error_when_no_key_set(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(NewsFetchError, match="FINNHUB_API_KEY"):
        get_market_news()


def test_get_market_news_parses_finnhub_response(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "fake-key")
    fake_response = MagicMock()
    fake_response.json.return_value = [
        {"headline": "Fed holds rates steady", "summary": "...", "source": "Reuters", "url": "https://example.com/fed", "datetime": 1755600000}
    ]
    fake_response.raise_for_status.return_value = None
    with patch("stockoptions.news.requests.get", return_value=fake_response) as mock_get:
        items = get_market_news(category="general")

    assert mock_get.call_args.kwargs["params"]["token"] == "fake-key"
    assert len(items) == 1
    assert items[0].title == "Fed holds rates steady"
    assert items[0].source == "finnhub"
    assert items[0].published_at == datetime.fromtimestamp(1755600000, tz=timezone.utc)


def test_get_market_news_raises_on_unexpected_response_shape(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "fake-key")
    fake_response = MagicMock()
    fake_response.json.return_value = {"error": "rate limited"}
    fake_response.raise_for_status.return_value = None
    with patch("stockoptions.news.requests.get", return_value=fake_response):
        with pytest.raises(NewsFetchError):
            get_market_news()


def test_search_mentions_filters_client_side_over_the_market_feed(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "fake-key")
    feed = [
        {"headline": "Trump announces new tariffs on steel", "summary": "", "source": "AP", "url": "u1", "datetime": 1},
        {"headline": "Widget Co quarterly report", "summary": "no mention here", "source": "AP", "url": "u2", "datetime": 2},
        {"headline": "Markets react", "summary": "Trump's comments moved futures overnight", "source": "AP", "url": "u3", "datetime": 3},
    ]
    fake_response = MagicMock()
    fake_response.json.return_value = feed
    fake_response.raise_for_status.return_value = None
    with patch("stockoptions.news.requests.get", return_value=fake_response):
        matches = search_mentions("Trump")

    assert len(matches) == 2
    assert all("trump" in (m.title + m.summary).lower() for m in matches)


@skip_unless_live
def test_get_ticker_news_live():
    items = get_ticker_news("AAPL", limit=5)
    assert len(items) > 0
    assert items[0].title
