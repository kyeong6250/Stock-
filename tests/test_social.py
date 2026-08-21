import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from stockoptions.social import SocialFetchError, get_truth_social_posts, get_truth_social_top_comments, get_x_posts

# ---------------------------------------------------------------------
# Truth Social (truthbrush)
# ---------------------------------------------------------------------


def test_get_truth_social_posts_raises_a_clear_error_when_truthbrush_isnt_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "truthbrush.api", None)
    with pytest.raises(SocialFetchError, match="truthbrush isn't installed"):
        get_truth_social_posts("realDonaldTrump")


def test_get_truth_social_posts_works_unauthenticated_for_a_public_account(monkeypatch):
    monkeypatch.delenv("TRUTHSOCIAL_USERNAME", raising=False)
    monkeypatch.delenv("TRUTHSOCIAL_PASSWORD", raising=False)

    fake_statuses = iter(
        [
            {"content": "<p>First post</p>", "created_at": "2026-08-20T13:40:09.736Z", "url": "https://truthsocial.com/@x/1"},
            {"content": "<p>Second post &amp; more</p>", "created_at": "2026-08-20T04:29:19.192Z", "url": "https://truthsocial.com/@x/2"},
        ]
    )
    fake_api = MagicMock()
    fake_api.pull_statuses.return_value = fake_statuses
    fake_api_cls = MagicMock(return_value=fake_api)

    fake_module = MagicMock()
    fake_module.Api = fake_api_cls
    fake_module.CFBlockException = type("CFBlockException", (Exception,), {})
    fake_module.GeoblockException = type("GeoblockException", (Exception,), {})
    fake_module.LoginErrorException = type("LoginErrorException", (Exception,), {})

    with patch.dict(sys.modules, {"truthbrush.api": fake_module}):
        posts = get_truth_social_posts("realDonaldTrump", limit=10)

    # require_auth should be False since no credentials were set -- the
    # whole point of this feature working for public accounts with zero setup.
    assert fake_api_cls.call_args.kwargs["require_auth"] is False
    assert len(posts) == 2
    assert posts[0].platform == "truth_social"
    assert posts[0].text == "First post"
    assert posts[0].posted_at == datetime(2026, 8, 20, 13, 40, 9, 736000, tzinfo=timezone.utc)
    assert posts[1].text == "Second post & more"


def test_get_truth_social_posts_uses_authenticated_mode_when_credentials_are_set(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_USERNAME", "someone")
    monkeypatch.setenv("TRUTHSOCIAL_PASSWORD", "hunter2")

    fake_api = MagicMock()
    fake_api.pull_statuses.return_value = iter([])
    fake_api_cls = MagicMock(return_value=fake_api)
    fake_module = MagicMock()
    fake_module.Api = fake_api_cls
    fake_module.CFBlockException = type("CFBlockException", (Exception,), {})
    fake_module.GeoblockException = type("GeoblockException", (Exception,), {})
    fake_module.LoginErrorException = type("LoginErrorException", (Exception,), {})

    with patch.dict(sys.modules, {"truthbrush.api": fake_module}):
        get_truth_social_posts("someuser")

    assert fake_api_cls.call_args.kwargs["require_auth"] is True
    assert fake_api_cls.call_args.kwargs["username"] == "someone"


def test_get_truth_social_posts_wraps_a_block_exception():
    fake_module = MagicMock()
    cf_block = type("CFBlockException", (Exception,), {})
    fake_module.CFBlockException = cf_block
    fake_module.GeoblockException = type("GeoblockException", (Exception,), {})
    fake_module.LoginErrorException = type("LoginErrorException", (Exception,), {})

    def raise_blocked(*a, **k):
        raise cf_block("blocked by Cloudflare")

    fake_api = MagicMock()
    fake_api.pull_statuses.side_effect = raise_blocked
    fake_module.Api = MagicMock(return_value=fake_api)

    with patch.dict(sys.modules, {"truthbrush.api": fake_module}):
        with pytest.raises(SocialFetchError, match="blocked"):
            get_truth_social_posts("realDonaldTrump")


def test_get_truth_social_posts_respects_limit():
    fake_module = MagicMock()
    fake_module.CFBlockException = type("CFBlockException", (Exception,), {})
    fake_module.GeoblockException = type("GeoblockException", (Exception,), {})
    fake_module.LoginErrorException = type("LoginErrorException", (Exception,), {})
    fake_api = MagicMock()
    fake_api.pull_statuses.return_value = iter([{"content": f"<p>post {i}</p>", "created_at": None, "url": None} for i in range(50)])
    fake_module.Api = MagicMock(return_value=fake_api)

    with patch.dict(sys.modules, {"truthbrush.api": fake_module}):
        posts = get_truth_social_posts("someuser", limit=3)
    assert len(posts) == 3


def test_get_truth_social_posts_raises_when_comments_requested_without_credentials(monkeypatch):
    monkeypatch.delenv("TRUTHSOCIAL_USERNAME", raising=False)
    monkeypatch.delenv("TRUTHSOCIAL_PASSWORD", raising=False)
    with pytest.raises(SocialFetchError, match="TRUTHSOCIAL_USERNAME"):
        get_truth_social_posts("realDonaldTrump", include_top_comments=True)


def test_get_truth_social_top_comments_raises_without_credentials(monkeypatch):
    monkeypatch.delenv("TRUTHSOCIAL_USERNAME", raising=False)
    monkeypatch.delenv("TRUTHSOCIAL_PASSWORD", raising=False)
    with pytest.raises(SocialFetchError, match="needs TRUTHSOCIAL_USERNAME"):
        get_truth_social_top_comments("12345")


def test_get_truth_social_top_comments_sorts_by_favourites_count_descending(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_USERNAME", "someone")
    monkeypatch.setenv("TRUTHSOCIAL_PASSWORD", "hunter2")

    raw_comments = [
        {"account": {"username": "low"}, "content": "<p>meh</p>", "favourites_count": 3, "url": "u1", "created_at": None},
        {"account": {"username": "top"}, "content": "<p>great point</p>", "favourites_count": 500, "url": "u2", "created_at": None},
        {"account": {"username": "mid"}, "content": "<p>ok</p>", "favourites_count": 40, "url": "u3", "created_at": None},
    ]
    fake_api = MagicMock()
    fake_api.pull_comments.return_value = iter(raw_comments)
    fake_module = MagicMock()
    fake_module.Api = MagicMock(return_value=fake_api)
    fake_module.CFBlockException = type("CFBlockException", (Exception,), {})
    fake_module.GeoblockException = type("GeoblockException", (Exception,), {})
    fake_module.LoginErrorException = type("LoginErrorException", (Exception,), {})

    with patch.dict(sys.modules, {"truthbrush.api": fake_module}):
        comments = get_truth_social_top_comments("12345", top_n=2)

    assert [c.author for c in comments] == ["top", "mid"]
    assert comments[0].favourites_count == 500


def test_get_truth_social_posts_attaches_top_comments_when_requested(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_USERNAME", "someone")
    monkeypatch.setenv("TRUTHSOCIAL_PASSWORD", "hunter2")

    fake_statuses = iter([{"id": 999, "content": "<p>main post</p>", "created_at": None, "url": "https://x/999"}])
    raw_comments = [{"account": {"username": "top"}, "content": "<p>reply</p>", "favourites_count": 10, "url": "u1", "created_at": None}]

    fake_api = MagicMock()
    fake_api.pull_statuses.return_value = fake_statuses
    fake_api.pull_comments.return_value = iter(raw_comments)
    fake_module = MagicMock()
    fake_module.Api = MagicMock(return_value=fake_api)
    fake_module.CFBlockException = type("CFBlockException", (Exception,), {})
    fake_module.GeoblockException = type("GeoblockException", (Exception,), {})
    fake_module.LoginErrorException = type("LoginErrorException", (Exception,), {})

    with patch.dict(sys.modules, {"truthbrush.api": fake_module}):
        posts = get_truth_social_posts("someuser", include_top_comments=True, top_comments_n=1)

    assert len(posts) == 1
    assert posts[0].id == "999"
    assert len(posts[0].top_comments) == 1
    assert posts[0].top_comments[0].author == "top"
    fake_api.pull_comments.assert_called_once_with("999", top_num=40)


# ---------------------------------------------------------------------
# X / Twitter (Nitter mirror)
# ---------------------------------------------------------------------


def test_get_x_posts_raises_when_no_instance_configured(monkeypatch):
    monkeypatch.delenv("NITTER_INSTANCE_URL", raising=False)
    with pytest.raises(SocialFetchError, match="NITTER_INSTANCE_URL"):
        get_x_posts("elonmusk")


def test_get_x_posts_detects_a_gate_page_disguised_as_200_ok(monkeypatch):
    monkeypatch.setenv("NITTER_INSTANCE_URL", "https://fake-instance.example")
    fake_response = MagicMock()
    fake_response.text = "<rss><channel><title>RSS reader not yet whitelisted!</title></channel></rss>"
    fake_response.raise_for_status.return_value = None
    with patch("stockoptions.social.requests.get", return_value=fake_response):
        with pytest.raises(SocialFetchError, match="doesn't actually work"):
            get_x_posts("elonmusk")


def test_get_x_posts_parses_a_real_looking_rss_feed(monkeypatch):
    monkeypatch.setenv("NITTER_INSTANCE_URL", "https://fake-instance.example")
    rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item><title>First post &amp; stuff</title><link>https://fake-instance.example/elonmusk/status/1</link>
<pubDate>Thu, 20 Aug 2026 13:40:09 GMT</pubDate></item>
<item><title>Second post</title><link>https://fake-instance.example/elonmusk/status/2</link>
<pubDate>Thu, 20 Aug 2026 04:29:19 GMT</pubDate></item>
</channel></rss>"""
    fake_response = MagicMock()
    fake_response.text = rss
    fake_response.raise_for_status.return_value = None
    with patch("stockoptions.social.requests.get", return_value=fake_response):
        posts = get_x_posts("elonmusk", limit=10)

    assert len(posts) == 2
    assert posts[0].platform == "x"
    assert posts[0].text == "First post & stuff"
    assert posts[0].url == "https://fake-instance.example/elonmusk/status/1"
    assert posts[0].posted_at is not None


def test_get_x_posts_raises_on_malformed_xml_despite_containing_item_tag(monkeypatch):
    monkeypatch.setenv("NITTER_INSTANCE_URL", "https://fake-instance.example")
    fake_response = MagicMock()
    fake_response.text = "<rss><channel><item>not closed properly"
    fake_response.raise_for_status.return_value = None
    with patch("stockoptions.social.requests.get", return_value=fake_response):
        with pytest.raises(SocialFetchError):
            get_x_posts("elonmusk")


def test_get_x_posts_respects_limit(monkeypatch):
    monkeypatch.setenv("NITTER_INSTANCE_URL", "https://fake-instance.example")
    items = "".join(f"<item><title>post {i}</title><link>https://x/{i}</link></item>" for i in range(10))
    fake_response = MagicMock()
    fake_response.text = f"<rss><channel>{items}</channel></rss>"
    fake_response.raise_for_status.return_value = None
    with patch("stockoptions.social.requests.get", return_value=fake_response):
        posts = get_x_posts("elonmusk", limit=4)
    assert len(posts) == 4
