# Stock- (stockoptions)

An options analysis toolkit built for fun, not as a stock predictor. It
does three genuinely different things, and is upfront about which of them
you should actually trust:

1. **Black-Scholes pricing, Greeks, and implied volatility** — exact math.
2. **Multi-leg strategy payoffs** (iron condors, spreads, strangles) —
   also exact math: given a position's legs, its P&L at expiration is
   arithmetic, not a guess.
3. **A directional up/down signal with a backtester** — the one genuinely
   weak piece, and it's built to say so out loud. See
   [Honest results](#honest-results-not-cherry-picked) below.
4. **A trade recommender** (`stockoptions predict`) built on top of (3) —
   a specific contract and position size, not just up/down. See
   [Trade recommender](#trade-recommender) below for exactly what part of
   this is real math (contract mechanics, Kelly-criterion sizing) versus
   still inheriting (3)'s weak directional edge.

No model here reliably predicts short-term stock direction. That's not a
limitation of this particular implementation — markets are close to
efficient at short horizons, well-documented in finance research, and if
a simple script could consistently beat that, it wouldn't stay simple (or
free) for long. This project treats that as the starting assumption, not
something to paper over with a confident-looking dashboard.

## Install

```sh
pip install -e ".[dev]"
# add extras for the pieces you actually want:
pip install -e ".[dev,web]"         # the dashboard (below)
pip install -e ".[dev,robinhood]"   # the `watchlist` command
pip install -e ".[dev,social]"      # Truth Social pulls (`influencer truth ...`)
pip install -e ".[dev,web,robinhood,social]"
```

Requires Python 3.11+.

## Web dashboard

```sh
stockoptions dashboard
```

Opens a local dashboard at `http://127.0.0.1:8000` (`--no-browser` to skip
auto-opening a tab, `--port` to change it): a ticker overview with a price
chart, the option chain with recomputed IV/Greeks, a strategy payoff
builder with a live P&L chart, and the backtest panel with the
strategy-vs-buy-and-hold equity curves. It's a thin UI over the exact
same tested Python functions the CLI calls — `/api/strategy`, for
instance, is `strategies.py`'s `max_profit`/`max_loss`/`breakevens`
wrapped in JSON, not a reimplementation. Runs locally only (it needs
live yfinance calls server-side); there's no hosted version.

Motion in the UI is deliberately restrained and functional-only —
stat values count up rather than snapping when they change, chart
lines draw in to confirm new data actually replaced the old render,
loading states shimmer instead of sitting blank — no decorative
bounce/glow/glassmorphism, and all of it degrades instantly and
correctly under `prefers-reduced-motion: reduce`.

Also applied a few concrete fintech/ML-dashboard UI conventions found
while researching this: every decision-critical number (current price,
a prediction, a backtest run) is paired with a "fetched HH:MM:SS"
freshness timestamp rather than looking permanently live — the
Bloomberg/LSEG-terminal convention, and a real disclosure here since
`data.py` caches quotes for up to 15 minutes locally. The Predict
panel's raw confidence percentage gets a plain-language chip (Low/
Moderate/High confidence, thresholds set from this project's own
backtest numbers rather than an arbitrary scale) alongside it, since a
bare "57.6%" doesn't on its own convey how strong a probabilistic model
output actually is. And feature importance (see Accuracy upgrades below)
renders as horizontal bars rather than a bare table, the standard
recommendation for this kind of ranked-weight display.

Sources: [fintech dashboard UI conventions (freshness timestamps, tabular figures, color reserved for financial state)](https://www.wildnetedge.com/blogs/fintech-ux-design-best-practices-for-financial-dashboards),
[ML/AI dashboard UX (plain-language confidence labels, bar-plot feature importance)](https://thefinch.design/ux-best-practices-ai-ml-data-visualization-dashboards/).

## Commands

```sh
# Exact payoff math for a position -- no market data needed, no prediction
stockoptions strategy iron-condor \
  --put-long-strike 90 --put-short-strike 95 \
  --call-short-strike 105 --call-long-strike 110 \
  --put-long-premium 1 --put-short-premium 2 \
  --call-short-premium 2 --call-long-premium 1
#            Iron Condor
# +-------------------------------+
# | metric       | value          |
# |--------------+----------------|
# | Net premium  | -2.00 (credit) |
# | Max profit   | 2.00           |
# | Max loss     | -3.00          |
# | Breakeven(s) | 93.00, 107.00  |
# +-------------------------------+

# Full option chain with IV/Greeks recomputed from real quotes (see why below)
stockoptions greeks AAPL --expiration 2026-08-28 --type call

# Is a ticker's options pricing in more or less movement than it's recently had?
stockoptions screen AAPL MSFT

# Backtest the directional signal against honest baselines
stockoptions backtest AAPL --period 2y --horizon 5

# A concrete trade suggestion: contract + Kelly-sized position count (see
# the Trade recommender section below before trusting the sizing)
stockoptions predict AAPL --account-size 100000 --risk-pct 2 --horizon 35

# Pull tickers from your Robinhood watchlist, then screen them (needs .env, see below)
stockoptions watchlist

# Recent headlines for a ticker (informational -- see News & influencer tracking below)
stockoptions news AAPL

# Recent posts from a public figure, read-only, unofficial (see the risk disclosure below)
stockoptions influencer truth realDonaldTrump
stockoptions influencer x elonmusk    # off by default -- see why in the section below
```

Also runnable as `python -m stockoptions <command>` if you didn't install
the console script.

## Honest results (not cherry-picked)

Running `stockoptions backtest AAPL --period 2y` against real market data
while building this produced:

| metric | value |
|---|---|
| Model accuracy | 48.2% |
| Majority-class baseline | 57.8% |
| Beats baseline? | **no** |
| Strategy total return | 24.4% |
| Buy & hold total return | 18.0% |

Notice the trap that result is designed to catch: a positive "strategy
total return" looks good in isolation and would make a convincing
screenshot. But the model's raw accuracy (48.2%, worse than a coin flip)
doesn't come close to the boring baseline of always guessing whichever
direction was more common in training data (57.8%) — AAPL just drifted
up over this window, and the strategy's return mostly reflects that
drift (amplified by the backtester's overlapping-position simplification,
see `backtest.py`), not a real predictive edge. This is exactly why
`backtest`'s output always shows accuracy next to the baseline instead of
a bare number: the bare number alone would have been actively misleading
here.

(This accuracy figure has moved twice, for two different reasons, and
both are worth sitting with if you're tempted to trust a stock-direction
backtest that looks decent. It originally read 56.3%; the train/test
leakage fix described below dropped it to 48.9% — a real bug, not
cosmetic: some of that original number was the model getting an unfair
peek across the train/test boundary. It now reads 48.2% after adding
three more technical indicators (see Accuracy upgrades below) — a small,
expected drift from changing the feature set, not a bug. Same
conclusion both times: doesn't beat the baseline.)

## Accuracy upgrades

Researched what would actually make this more accurate before building
it (see sources at the bottom of this section) rather than guessing.
Five concrete gaps, in order of how much they mattered:

**1. Train/test label leakage at the split boundary (real bug, changed
the actual conclusion).** A label for day *i* is computed by looking
`horizon_days` ahead. Without a gap, training rows within `horizon_days`
of the train/test split have labels that peek across the boundary into
the test set — precisely the leakage pattern the research flagged:
*"a moving average calculated at the end of the in-sample period depends
on prices that extend into the out-of-sample period... creates data
leakage at the boundary."* `backtest.py` now purges the last
`horizon_days` training rows before the split. This is the fix that
dropped the AAPL result above from 56.3% to 48.9% — the leaked rows were
inflating apparent accuracy.

**2. Real US equity options are American-style; this project only ever
priced/solved-IV with European Black-Scholes.** That's a known,
textbook-documented source of error — an American option's early-exercise
right has real value Black-Scholes can't price in. `binomial.py` adds a
Cox-Ross-Rubinstein tree with early exercise checked at every node, and
is now the default pricing model `data.py` uses to recompute IV and
Greeks. Cross-validated against `blackscholes.py`: an American call on a
non-dividend stock converges to the Black-Scholes price (the textbook
"never optimal to early-exercise" result), while American puts and
dividend-paying calls price strictly higher, as they should.

**3. `r` was a hardcoded 0.05 guess everywhere.** `rates.py` fetches the
real, current Treasury yield curve and interpolates a rate matched to
each option's own time-to-expiration, per standard practice confirmed
while researching this (maturity-match the risk-free rate to the
option's term). Live-checked while building this that short and long
rates can differ by more than a full point (13-week bill at 3.70% vs.
30-year bond at 5.19%) — a single flat constant was never a good
approximation.

**4. `q` (dividend yield) defaulted to 0 for every ticker.**
`get_dividend_yield()` fetches the real trailing dividend yield instead.
Also had to sort out a units mismatch on yfinance's side: its
`dividendYield` field turned out to already be a percentage number (2.39
meaning 2.39%), while `trailingAnnualDividendYield` is the same figure
already expressed as a decimal (0.0239) — confirmed by cross-checking
`trailingAnnualDividendRate / price` against both fields across five
tickers, including a non-dividend payer, before trusting either.

Feature scaling (`StandardScaler` in `signals.py`'s training pipeline)
was also added, since the technical-indicator features live on very
different scales (RSI spans 0-100; most others are small decimals) and
logistic regression's regularization implicitly under-weights
smaller-magnitude features for reasons that have nothing to do with
their actual predictive value.

**5. More technical indicators, and a model comparison that didn't go
the way the sources suggested it would.** Looked at what other
open-source technical-indicator classifiers use and added three: MACD
histogram, Bollinger %B, and Average True Range (`signals.py`) — one
project's own reported feature-importance ranking specifically called
out Bollinger Bands as substantially more informative than its other
features, which is part of why it made the cut here. Also added
`RandomForestClassifier` as a selectable alternative to logistic
regression (`train(..., model_type="random_forest")`, same for
`backtest()`/`predict()`), since multiple sources reported ensembles
consistently outperforming plain logistic regression on this kind of
task. Tested that claim rather than taking it on faith: ran both model
types head-to-head on AAPL, MSFT, and TSLA (2y history, 5-day horizon).

| ticker | logistic accuracy | random forest accuracy | baseline | which beat baseline |
|---|---|---|---|---|
| AAPL | 48.1% | 57.0% | 57.8% | neither |
| MSFT | 45.9% | 44.4% | 48.1% | neither |
| TSLA | 49.6% | 36.3% | 42.2% | **logistic only** |

No consistent winner — random forest is meaningfully better on AAPL,
meaningfully *worse* on TSLA, and close-but-slightly-worse on MSFT, and
logistic regression is the only one of the two that beat its baseline
anywhere in this sample (TSLA). That's not the sources' "ensembles
consistently outperform" claim holding up on this project's own data,
with its own features, over this particular sample of tickers — three
tickers is too small a sample to draw a general conclusion from either
way, which is itself the point: logistic regression stays the default
because switching wasn't a demonstrated improvement, not because it was
demonstrated to be better. `model_type="random_forest"` is left in as an
option for anyone who wants to try it on their own tickers, not removed
just because this particular small comparison didn't favor it.

The Bollinger-importance claim didn't hold up either, on its own terms.
`TrainedModel.feature_importance()` exposes exactly what each trained
model weights (shown on `stockoptions backtest`'s CLI output and the
dashboard's Backtest panel, as a bar chart per the UI research below).
Running it on AAPL's actual training window: `macd_hist_norm` and
`rsi_14` come out on top for both model types, with `bollinger_pctb`
solidly mid-pack (roughly 8% of total weight, versus MACD's ~37%) — not
"substantially more informative than other features" the way the source
study reported for its own data. Kept anyway: MACD/ATR turning out to
matter more here is still a genuine finding this project wouldn't have
without adding all three, and one ticker's importance ranking, like the
three-ticker model comparison above, isn't a large enough sample to
conclude Bollinger Bands are never useful — only that this particular
claim didn't transfer to this particular model on this particular data.

Sources: [risk-free rate maturity matching](https://fastercapital.com/content/The-Role-of-Risk-Free-Rates-in-Black-Scholes-Pricing.html),
[binomial vs. Black-Scholes for American options](https://mbrenndoerfer.com/writing/binomial-tree-option-pricing-cox-ross-rubinstein),
[walk-forward validation and label leakage](https://blog.quantinsti.com/walk-forward-optimization-python-xgboost-stock-prediction/),
[technical indicators for ML stock prediction (MACD/Bollinger/RSI/etc.)](https://github.com/alisonmitchell/Stock-Prediction),
[random forest for stock direction, incl. Bollinger feature importance](https://usman-haider.medium.com/predicting-stock-market-movement-with-technical-indicators-and-random-forest-step-by-step-python-2d797d5c7b24),
[gradient-boosted ensembles outperforming logistic regression on stock trend prediction](https://doaj.org/article/fbff47aeb10a4ee6927e32efeec21ceb).

## Trade recommender

```sh
stockoptions predict AAPL --account-size 100000 --risk-pct 2 --horizon 35 --delta 0.35
```

Also on the dashboard's "Predict" panel, with a price-projection chart.
This answers the question the rest of the project deliberately stops
short of: not just "is this ticker's directional signal up or down," but
"which specific contract, and how many." Two genuinely different kinds
of claim get made here, and it matters which is which:

**Real, reliable regardless of the signal's accuracy:**
- **Contract selection** is standard practitioner mechanics: the
  expiration nearest a 30-45 day target (the conventional "theta high,
  gamma low" sweet spot for a multi-week directional hold — confirmed
  while researching this: front-month options have aggressive theta
  curves, while 30-45 DTE keeps decay roughly linear until the last
  2-3 weeks), and the strike whose recomputed delta is closest to a
  target (0.30-0.40 delta is the standard directional-trader range,
  balancing cost against probability of finishing ITM).
- **Position sizing is a real Kelly-criterion calculation**
  (`f* = win_rate - (1 - win_rate) / payout_ratio`, half-Kelly applied by
  default since full Kelly is known to be too aggressive for real
  drawdowns), fed an *empirical* edge estimate: rather than assuming a
  payout ratio, `recommend.py` replays `backtest.py`'s own purged-gap,
  no-lookahead test-period rows through the chosen contract's moneyness
  and current IV via the binomial pricing model, to see what buying this
  shape of contract would actually have paid off historically whenever
  the model predicted today's exact direction. The one disclosed
  approximation: no free source of historical implied volatility exists,
  so *today's* IV stands in for every historical date's IV in that
  replay.
- The **price-projection chart** is the standard "expected move" cone
  used across options platforms (Barchart, tastytrade, projectoption,
  etc.): `expected_move(t) = price × IV × √(t / 365)`, giving a
  1-standard-deviation band (~68%) and 2-standard-deviation band (~95%)
  that widens with time. It's a probability *range* derived from the
  market's own priced-in volatility, not a prediction of where the price
  will land — shown as a widening cone specifically so the uncertainty
  stays visible instead of implying false precision.

**Still inherits (3)'s weak edge, and is built to say so:** the
*direction* being sized is still today's live call from the same
directional classifier `backtest.py` already shows, honestly, often
failing to beat a majority-class baseline. Real Kelly math fed a weak or
negative edge does the right thing on its own — it recommends risking
little or nothing, which is exactly what you'll see on tickers/horizons
where the signal has no real edge, not a separate safety check
second-guessing the math. `TradeRecommendation.warnings` spells out
*why* whenever that happens: backtest accuracy not beating its baseline,
too few historical instances matching today's predicted direction to
trust the sizing (`n` is always shown), a Kelly fraction at or below
zero, or a recommended size that rounds down to 0 contracts at your
account size and risk cap. None of that is the tool being unhelpful —
it's the same "48.9% accuracy, below a coin flip" honesty from the
Honest results section above, applied to position sizing instead of a
bare backtest number.

Sources: [delta selection for directional trades](https://pomegra.io/learn/library/track-e-trading-risk/options-beginners/chapter-11-choosing-strikes-and-expiries/delta-selection-guide),
[DTE selection and theta decay](https://www.daystoexpiry.com/blog/theta-decay-dte-guide),
[Kelly criterion for position sizing](https://longbridge.com/en/academy/options/blog/options-position-sizing-kelly-criterion-explained-100160),
[expected-move probability cones](https://gocharting.com/docs/options-desk/options-probability-cone).

## Why recompute IV instead of trusting yfinance's own column?

Pull a deep ITM options chain from yfinance and check its
`impliedVolatility` column — you'll see values like 800%, obvious noise
from stale/illiquid quotes, not a real market view. Rather than trust
that, `data.py` recomputes IV per contract from its own bid/ask midpoint,
using the American binomial model by default (see Accuracy upgrades
above), and skips (as NaN) any contract whose price doesn't correspond to
a valid IV rather than letting one bad row poison a whole chain scan.

Similarly, `screen`'s IV/HV comparison is deliberately *not* called an
"IV rank": a true IV rank needs a historical time series of implied
vol, which free `yfinance` data doesn't provide. Comparing current IV to
trailing realized volatility instead is a standard practitioner proxy —
still useful, but the README (and the CLI's own output) says proxy, not
rank.

## Robinhood integration (optional, real risk)

`stockoptions watchlist` pulls tickers from your actual Robinhood
watchlist via `robin_stocks`. Before using it:

- Robinhood has **no official public API**. `robin_stocks` reverse-engineers
  their private app API, which violates Robinhood's Terms of Service.
- This project only ever *calls* read-style functions (login, and
  `get_*`/`build_*` on the account module) — never anything from
  `robin_stocks.robinhood.orders` (order placement/cancellation) or the
  watchlist *write* functions. See `robinhood.py`'s docstring and
  `tests/test_robinhood.py` for exactly what guarantee that is and isn't
  (an earlier, stronger-sounding claim about this turned out to be false
  the first time it was actually tested — worth reading if you're
  trusting this with your real account).
- **Even read-only automated access can still get an account flagged or
  suspended.** That risk isn't eliminated by this code being read-only —
  it's inherent to using an unofficial API at all.

Setup: `cp .env.example .env`, fill in `ROBINHOOD_USERNAME` /
`ROBINHOOD_PASSWORD` (and optionally `ROBINHOOD_MFA_CODE`). `.env` is
gitignored. Every other command works with zero Robinhood credentials —
you just type tickers in yourself.

## News & influencer tracking

Recent headlines and public figures' recent posts, on the dashboard's
"News & catalysts" panel and via `stockoptions news`/`stockoptions
influencer`. This is **informational only** -- it doesn't feed
`backtest.py`'s directional signal. Turning headlines/posts into a
sentiment feature is a real technique with a real (and mixed, easy-to-
overfit) track record in the research, and doing it honestly needs its
own walk-forward validation work this project hasn't done. Bolting it
onto the existing model without that would be exactly the kind of
confident-looking-but-unvalidated number this README keeps arguing
against elsewhere (see Honest results above). So for now: it's here for
a human to read, not for the model to use.

**Ticker news** (`news.py`): free, no setup. Pulls straight from Yahoo
Finance via yfinance (same dependency `data.py` already uses). Add a free
[Finnhub](https://finnhub.io) API key (`FINNHUB_API_KEY` in `.env`, see
`.env.example`) for `get_market_news()`/`search_mentions()` -- broader,
ticker-agnostic market news and a simple client-side keyword filter over
it (a substring filter, not a real search index -- Finnhub's free tier
doesn't offer full-text search on this endpoint).

**Influencer watch** (`social.py`): unofficial, read-only pulls from a
small, fixed watchlist (currently Trump on Truth Social, Musk on X) --
deliberately not a free-text field, so nothing typed into the dashboard
ever reaches these scrapers. Real risk, same category as the Robinhood
integration above, read before using:

- **Truth Social** works today with **no login required** for reading a
  public account -- live-verified while building this
  (`Api(require_auth=False).pull_statuses("realDonaldTrump")` returned
  real, current posts). Uses [`truthbrush`](https://github.com/stanfordio/truthbrush)
  (Stanford Internet Observatory), which calls Truth Social's own private
  app API -- Truth Social has no self-serve public API, and whether
  truthbrush's own "publicly accessible" framing makes this
  ToS-compliant is a real, unresolved question this project doesn't
  settle for you. Optional `TRUTHSOCIAL_USERNAME`/`PASSWORD` in `.env`
  switch to authenticated mode for higher rate limits (default pull is
  now 20 posts, up from an earlier, too-thin 10) -- and, only with those
  credentials set, also unlock each post's top few comments by like
  count (`--comments` on the CLI, shown automatically on the dashboard
  when credentials are present). Reading a public account's own posts
  needs no login; reading *comments* on a post does, even a public one --
  a real asymmetry in truthbrush's underlying API, not a design choice
  here. truthbrush itself only fetches replies oldest-first server-side,
  so "top-rated" is this project's own client-side sort by like count
  over a batch of recent replies, not a guarantee of surfacing the single
  most-liked reply on a post with thousands of comments.
- **X (Twitter)** is **off by default and not expected to work**. X's
  official API has no meaningful free tier, and free scraping mirrors
  (Nitter) have almost entirely collapsed under X's anti-scraping
  lockdown. Live-checked against five public Nitter instances while
  building this: three errored outright (HTTP 429/403/500), one had a
  dead DNS entry, and the one that returned HTTP 200 (`xcancel.com`)
  turned out to be serving an "RSS reader not yet whitelisted" gate page,
  not real posts -- a false-positive worth calling out on its own: a
  naive "did the request succeed?" check would have shipped a feature
  that silently returns garbage forever. `get_x_posts()` detects that gate
  and raises instead of pretending it worked; the dashboard shows X as
  "unavailable" with the real reason rather than an empty list. Set
  `NITTER_INSTANCE_URL` only if you have a specific instance you've
  personally confirmed still returns real posts.

## What it does and doesn't cover

Covers: Black-Scholes pricing/Greeks/IV, multi-leg strategy payoffs
(vertical spreads, strangles, straddles, iron condors, covered calls),
IV-vs-realized-vol screening, a backtested (honestly, against baselines)
directional signal, a Kelly-sized trade recommender built on top of that
signal (see Trade recommender above), and an informational news/
influencer panel (see News & influencer tracking above -- explicitly not
part of the signal).

Doesn't cover: Special Monthly Compensation-style exotic payoffs, real
order execution, portfolio-level margin/risk, or anything requiring a
paid data feed. Put-call parity / arbitrage-style checks aren't
implemented as a "free money finder" — any genuine violation gets closed
by market makers before a retail script would ever see it.

## Architecture

```
blackscholes.py   European pricing, Greeks, implied volatility (pure math)
binomial.py       American pricing/Greeks/IV via a CRR tree (pure math)
rates.py          real, maturity-matched Treasury risk-free rate
volatility.py     realized vol, IV rank/percentile, IV skew z-score (pure)
strategies.py     multi-leg payoff/max-profit/max-loss/breakevens (pure)
data.py           yfinance wrapper w/ local caching + IV/Greeks recompute
analysis.py       screen_ticker() -- shared by the CLI and the dashboard
signals.py        technical features + scaled logistic regression classifier
backtest.py       purged-gap train/test backtest vs. honest baselines
recommend.py      contract selection + empirical-Kelly position sizing + expected-move cone
robinhood.py      read-only watchlist/positions pull (optional)
news.py           ticker/market news, informational only (yfinance + optional Finnhub)
social.py         unofficial read-only influencer post pulls (optional, real ToS risk)
cli.py            argparse subcommands, rich tables, clean error messages
web/app.py        FastAPI wrapper exposing the same functions as JSON (optional)
web/static/       vanilla HTML/CSS/JS dashboard, zero frontend dependencies
```

The pure modules (`blackscholes`, `volatility`, `strategies`) have no
network or I/O dependency at all, so they're fully unit-testable with
synthetic data — that's most of the test suite. `data.py`'s network calls
are exercised against the real yfinance API too (gated behind
`STOCKOPTIONS_LIVE_TESTS=1` so CI doesn't depend on a third-party service
being up), not just mocked.

## Development

```sh
pip install -e ".[dev,robinhood]"
pytest -v                                    # offline tests (CI-safe, fast)
STOCKOPTIONS_LIVE_TESTS=1 pytest -v           # also hits the real yfinance API
```

## Inspired by

Built after looking at several existing options-analysis projects for
technique ideas (not code): [OptionArbitrage](https://github.com/vincentdamato/OptionArbitrage)'s
put-call-parity/IV-skew-z-score approach, [Stock-Options-Analysis-Tool](https://github.com/guccipepito/Stock-Options-Analysis-Tool)'s
Black-Scholes/yfinance combo, and [mirajgodha/options](https://github.com/mirajgodha/options)'s
multi-leg strategy payoff calculator, which is where the idea for
`strategies.py` — the one piece of this whole project that involves no
prediction at all — came from. [alisonmitchell/Stock-Prediction](https://github.com/alisonmitchell/Stock-Prediction)'s
list of technical indicators (MACD, Bollinger Bands, Stochastic
Oscillator, MFI, ROC, OBV) is where the idea for the three newer
`signals.py` features came from, though this project only adopted the
three that seemed most complementary to the original five rather than
all of them.

## License

MIT — see [LICENSE](LICENSE). Not affiliated with Robinhood, Yahoo
Finance, or any of the projects linked above. Not investment advice.
