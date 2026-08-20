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

No model here reliably predicts short-term stock direction. That's not a
limitation of this particular implementation — markets are close to
efficient at short horizons, well-documented in finance research, and if
a simple script could consistently beat that, it wouldn't stay simple (or
free) for long. This project treats that as the starting assumption, not
something to paper over with a confident-looking dashboard.

## Install

```sh
pip install -e ".[dev]"
# add the robinhood extra only if you're using the `watchlist` command:
pip install -e ".[dev,robinhood]"
```

Requires Python 3.11+.

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

# Pull tickers from your Robinhood watchlist, then screen them (needs .env, see below)
stockoptions watchlist
```

Also runnable as `python -m stockoptions <command>` if you didn't install
the console script.

## Honest results (not cherry-picked)

Running `stockoptions backtest AAPL --period 2y` against real market data
while building this produced:

| metric | value |
|---|---|
| Model accuracy | 56.3% |
| Majority-class baseline | 57.8% |
| Beats baseline? | **no** |
| Strategy total return | 89.4% |
| Buy & hold total return | 17.3% |

Notice the trap that result is designed to catch: "89.4% strategy return"
looks great in isolation, and would make a convincing screenshot. But the
model's raw accuracy (56.3%) didn't even beat the boring baseline of
always guessing whichever direction was more common in training data
(57.8%) — AAPL just drifted up a lot over this window, and the strategy's
return mostly reflects that drift (amplified by the backtester's
overlapping-position simplification, see `backtest.py`), not a real
predictive edge. This is exactly why `backtest`'s output always shows
accuracy next to the baseline instead of a bare number: the bare number
alone would have been actively misleading here.

## Why recompute IV instead of trusting yfinance's own column?

Pull a deep ITM options chain from yfinance and check its
`impliedVolatility` column — you'll see values like 800%, obvious noise
from stale/illiquid quotes, not a real market view. Rather than trust
that, `data.py` recomputes IV per contract from its own bid/ask midpoint
via the Black-Scholes solver in `blackscholes.py`, and skips (as NaN)
any contract whose price doesn't correspond to a valid IV rather than
letting one bad row poison a whole chain scan.

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

## What it does and doesn't cover

Covers: Black-Scholes pricing/Greeks/IV, multi-leg strategy payoffs
(vertical spreads, strangles, straddles, iron condors, covered calls),
IV-vs-realized-vol screening, and a backtested (honestly, against
baselines) directional signal.

Doesn't cover: Special Monthly Compensation-style exotic payoffs, real
order execution, portfolio-level margin/risk, or anything requiring a
paid data feed. Put-call parity / arbitrage-style checks aren't
implemented as a "free money finder" — any genuine violation gets closed
by market makers before a retail script would ever see it.

## Architecture

```
blackscholes.py   pricing, Greeks, implied volatility (pure math, no I/O)
volatility.py     realized vol, IV rank/percentile, IV skew z-score (pure)
strategies.py     multi-leg payoff/max-profit/max-loss/breakevens (pure)
data.py           yfinance wrapper w/ local caching + IV/Greeks recompute
signals.py        technical features + logistic regression classifier
backtest.py       chronological train/test backtest vs. honest baselines
robinhood.py      read-only watchlist/positions pull (optional)
cli.py            argparse subcommands, rich tables, clean error messages
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
prediction at all — came from.

## License

MIT — see [LICENSE](LICENSE). Not affiliated with Robinhood, Yahoo
Finance, or any of the projects linked above. Not investment advice.
