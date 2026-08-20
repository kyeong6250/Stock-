"""Unofficial social-media pulls for named public figures whose posts are
known to move markets (Trump, Musk, etc.) -- separate from news.py
because, unlike the news APIs there, everything here is optional,
unofficial, and carries real platform-ToS risk. Same risk category as
robinhood.py; read that module's docstring too if you haven't.

Truth Social (`get_truth_social_posts`): uses `truthbrush` (Stanford
Internet Observatory), which calls Truth Social's own private app API the
same way the official app does -- Truth Social has no self-serve public
API. Whether "publicly accessible" (truthbrush's own framing) makes this
ToS-compliant is a real, unresolved question this project doesn't settle
for you. Live-verified while building this (2026-08-20): reading a public
figure's posts works with *no login at all* (`require_auth=False`) --
`Api(require_auth=False).pull_statuses("realDonaldTrump")` returned real,
current posts in testing. TRUTHSOCIAL_USERNAME/PASSWORD (see .env.example)
switches to authenticated mode for higher rate limits; not required for
the common case of just reading a public account.

X (formerly Twitter) (`get_x_posts`): X's official API has no meaningful
free tier as of this writing, and free scraping mirrors (Nitter) have
almost entirely collapsed under X's anti-scraping lockdown. Live-checked
against five public Nitter instances while building this: three errored
outright (429/403/500), one had a dead DNS entry, and the one that
returned HTTP 200 (xcancel.com) turned out to be serving an "RSS reader
not yet whitelisted" gate page, not real posts -- a false-positive worth
flagging since a naive "did the request succeed?" check would have
shipped a feature that silently returns garbage forever. get_x_posts()
detects known gate-page markers and an absent <item> feed and raises
instead of pretending it worked. There's no bundled default instance
because none of the ones checked actually work; point
NITTER_INSTANCE_URL at a self-hosted or currently-working instance if you
have one.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import requests
from defusedxml import ElementTree
from defusedxml.ElementTree import ParseError
from dotenv import load_dotenv


class SocialFetchError(RuntimeError):
    """Raised when a social post pull fails, is blocked, or the source isn't configured/working."""


@dataclass
class SocialPost:
    platform: str  # "truth_social" | "x"
    author: str
    text: str
    url: str
    posted_at: datetime | None


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    text = _TAG_RE.sub("", html or "")
    return text.replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"').strip()


def get_truth_social_posts(username: str, limit: int = 10) -> list[SocialPost]:
    """Recent posts from a public Truth Social account, e.g. "realDonaldTrump".
    Works with no credentials for reading a public account; set
    TRUTHSOCIAL_USERNAME/PASSWORD in .env for authenticated (higher-limit)
    access. Raises SocialFetchError on any truthbrush failure (blocked,
    geoblocked, bad login, network) rather than letting a bare exception
    from a third-party scraper surface to callers."""
    try:
        from truthbrush.api import Api, CFBlockException, GeoblockException, LoginErrorException
    except ImportError as exc:
        raise SocialFetchError("truthbrush isn't installed -- run `pip install -e '.[social]'`") from exc

    load_dotenv()
    ts_user = os.environ.get("TRUTHSOCIAL_USERNAME")
    ts_pass = os.environ.get("TRUTHSOCIAL_PASSWORD")
    api = Api(username=ts_user, password=ts_pass, require_auth=bool(ts_user and ts_pass))

    posts = []
    try:
        for raw in api.pull_statuses(username, replies=False):
            if len(posts) >= limit:
                break
            posts.append(
                SocialPost(
                    platform="truth_social",
                    author=username,
                    text=_strip_html(raw.get("content", "")),
                    url=raw.get("url") or f"https://truthsocial.com/@{username}",
                    posted_at=_parse_iso8601(raw.get("created_at")),
                )
            )
    except (CFBlockException, GeoblockException, LoginErrorException) as exc:
        raise SocialFetchError(f"Truth Social blocked this request ({type(exc).__name__}): {exc}") from exc
    except SocialFetchError:
        raise
    except Exception as exc:  # truthbrush doesn't expose one common base exception for every failure mode
        raise SocialFetchError(f"failed to pull Truth Social posts for {username!r}: {exc}") from exc
    return posts


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# Substrings seen in the gate/error pages some "working" Nitter mirrors
# serve with a misleading HTTP 200 (see module docstring). Extend this if
# a new instance turns out to have its own gate page -- checking for a
# real <item> below already catches most such cases, but this catches the
# "empty feed dressed up as content" case too.
_GATE_MARKERS = ("rss reader not", "not yet whitelisted", "access denied", "please enable js", "checking your browser")


def get_x_posts(username: str, limit: int = 10) -> list[SocialPost]:
    """Recent posts from an X/Twitter account, via a Nitter-style RSS
    mirror. Off by default -- set NITTER_INSTANCE_URL to a working
    instance (see this module's docstring for why none is bundled: every
    public instance checked while building this was dead, rate-limited,
    or gate-walled)."""
    load_dotenv()
    instance = os.environ.get("NITTER_INSTANCE_URL")
    if not instance:
        raise SocialFetchError(
            "NITTER_INSTANCE_URL isn't set. X/Twitter has no working free scraping route as of "
            "this writing (see social.py's module docstring for what was actually tested) -- set "
            "this only if you have a specific instance you've confirmed still works."
        )

    url = urljoin(instance.rstrip("/") + "/", f"{username}/rss")
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise SocialFetchError(f"request to {instance} failed: {exc}") from exc

    body = resp.text
    lowered = body.lower()
    if any(marker in lowered for marker in _GATE_MARKERS) or "<item>" not in body:
        raise SocialFetchError(
            f"{instance} returned HTTP 200 but no real post data (looks like a gate/error page, "
            "not an RSS feed of posts) -- this instance doesn't actually work right now."
        )

    try:
        # defusedxml, not stdlib ElementTree: this is untrusted XML from a
        # user-configured, possibly-arbitrary third-party URL, so it needs
        # protection against XXE/billion-laughs, not just a happy-path parse.
        root = ElementTree.fromstring(body)
    except ParseError as exc:
        raise SocialFetchError(f"{instance} returned a 200 response that isn't valid RSS/XML: {exc}") from exc

    posts = []
    for item in root.findall(".//item")[:limit]:
        title = item.findtext("title") or ""
        link = item.findtext("link") or url
        pub_date = item.findtext("pubDate")
        posted_at = None
        if pub_date:
            try:
                posted_at = parsedate_to_datetime(pub_date)
            except (TypeError, ValueError):
                posted_at = None
        posts.append(
            SocialPost(
                platform="x",
                author=username,
                text=_strip_html(title),
                url=link.strip(),
                posted_at=posted_at,
            )
        )
    return posts
