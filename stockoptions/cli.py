"""Command-line interface tying the toolkit together. Errors are caught
here and printed as one clean line, never a raw traceback -- see _fail().
"""

import argparse
import math
import sys

from rich.console import Console
from rich.table import Table

from stockoptions import __version__
from stockoptions.analysis import screen_ticker
from stockoptions.backtest import backtest
from stockoptions.blackscholes import NoArbitrageViolation
from stockoptions.data import (
    TickerNotFoundError,
    enrich_chain_with_iv_and_greeks,
    get_current_price,
    get_dividend_yield,
    get_option_chain,
    get_price_history,
    years_to_expiration,
)
from stockoptions.rates import get_yield_curve, risk_free_rate
from stockoptions.strategies import breakevens, iron_condor, max_loss, max_profit, strangle, vertical_spread

console = Console()

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_DATA_ERROR = 3

DISCLAIMER = (
    "[dim]For education/fun, not investment advice. No model reliably predicts short-term stock "
    "direction -- see the backtest baseline before trusting any 'signal' here.[/dim]"
)


def _fail(message: str, exit_code: int) -> None:
    console.print(f"[bold red]stockoptions: error:[/bold red] {message}")
    sys.exit(exit_code)


def cmd_greeks(args: argparse.Namespace) -> None:
    S = get_current_price(args.ticker)
    calls, puts = get_option_chain(args.ticker, args.expiration)
    chain = calls if args.type == "call" else puts

    T = years_to_expiration(args.expiration)
    r = risk_free_rate(T)  # real, maturity-matched Treasury yield -- not a hardcoded guess
    q = get_dividend_yield(args.ticker)
    enriched = enrich_chain_with_iv_and_greeks(chain, S, args.expiration, args.type, r, q)

    table = Table(title=f"{args.ticker} {args.type}s, exp {args.expiration}  (spot ${S:.2f}, r={r:.2%}, q={q:.2%})")
    for col in ["strike", "lastPrice", "my_iv", "delta", "gamma", "vega", "theta"]:
        table.add_column(col)
    for _, row in enriched.iterrows():
        if math.isnan(row["my_iv"]):
            continue
        table.add_row(
            f"{row['strike']:.2f}",
            f"{row['lastPrice']:.2f}",
            f"{row['my_iv']:.1%}",
            f"{row['delta']:.3f}",
            f"{row['gamma']:.4f}",
            f"{row['vega']:.3f}",
            f"{row['theta']:.3f}",
        )
    console.print(table)
    console.print(DISCLAIMER)


def cmd_screen(args: argparse.Namespace) -> None:
    table = Table(title="IV vs. realized volatility screen")
    for col in ["ticker", "price", "ATM IV (~30d)", "30d realized vol", "IV/HV", "read"]:
        table.add_column(col)

    yield_curve = get_yield_curve()  # fetch once, reuse (maturity-interpolated) for every ticker below

    for ticker in args.tickers:
        overview = screen_ticker(ticker, yield_curve)
        colored_read = {
            "rich": "[green]rich[/green]",
            "cheap": "[cyan]cheap[/cyan]",
            "in line": "in line",
        }[overview.read]
        table.add_row(
            ticker,
            f"${overview.price:.2f}",
            f"{overview.atm_iv:.1%}",
            f"{overview.realized_vol_30d:.1%}",
            f"{overview.iv_hv_ratio:.2f}",
            colored_read,
        )

    console.print(table)
    console.print(
        "[dim]rich = options pricing in more move than recent history; "
        "cheap = pricing in less move than recent history.[/dim]"
    )
    console.print(
        "[dim]IV/HV, not a true IV rank: yfinance doesn't provide a historical implied-vol series to "
        "rank against, so this compares current implied vol to trailing realized vol instead -- a "
        "standard practitioner proxy, but a proxy.[/dim]"
    )
    console.print(DISCLAIMER)


def cmd_backtest(args: argparse.Namespace) -> None:
    history = get_price_history(args.ticker, period=args.period)
    result = backtest(history, horizon_days=args.horizon)

    table = Table(title=f"{args.ticker} backtest ({args.horizon}-day horizon, {args.period} history)")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("Train / test samples", f"{result.n_train_samples} / {result.n_test_samples}")
    table.add_row("Model accuracy", f"{result.accuracy:.1%}")
    table.add_row("Majority-class baseline", f"{result.majority_baseline_accuracy:.1%}")
    table.add_row("Beats baseline?", "yes" if result.beats_baseline else "no")
    table.add_row("Strategy total return", f"{result.strategy_total_return:.1%}")
    table.add_row("Buy & hold total return", f"{result.buy_and_hold_total_return:.1%}")
    table.add_row("Strategy Sharpe", f"{result.strategy_sharpe:.2f}")
    table.add_row("Strategy max drawdown", f"{result.strategy_max_drawdown:.1%}")
    console.print(table)
    if not result.beats_baseline:
        console.print("[yellow]This signal did not beat the majority-class baseline on this ticker/period.[/yellow]")
    console.print(DISCLAIMER)


def cmd_strategy(args: argparse.Namespace) -> None:
    if args.kind == "vertical":
        strat = vertical_spread(args.option_type, args.long_strike, args.short_strike, args.long_premium, args.short_premium)
    elif args.kind == "strangle":
        strat = strangle(args.call_strike, args.put_strike, args.call_premium, args.put_premium, short=not args.long)
    else:  # iron-condor
        strat = iron_condor(
            args.put_long_strike, args.put_short_strike, args.call_short_strike, args.call_long_strike,
            args.put_long_premium, args.put_short_premium, args.call_short_premium, args.call_long_premium,
        )

    profit = max_profit(strat)
    loss = max_loss(strat)
    bes = breakevens(strat)

    table = Table(title=strat.name)
    table.add_column("metric")
    table.add_column("value")
    table.add_row("Net premium", f"{strat.net_premium():.2f} ({'debit' if strat.net_premium() >= 0 else 'credit'})")
    table.add_row("Max profit", "unlimited" if profit == float("inf") else f"{profit:.2f}")
    table.add_row("Max loss", "unlimited" if loss == float("-inf") else f"{loss:.2f}")
    table.add_row("Breakeven(s)", ", ".join(f"{b:.2f}" for b in bes) if bes else "none found in scan range")
    console.print(table)
    console.print("[dim]Pure payoff math for the legs you gave it -- no prediction involved here.[/dim]")


def cmd_watchlist(args: argparse.Namespace) -> None:
    from stockoptions.robinhood import RobinhoodAuthError, connect, disconnect, get_watchlist_tickers

    try:
        connect()
        tickers = get_watchlist_tickers(args.name)
    except RobinhoodAuthError as exc:
        _fail(str(exc), EXIT_BAD_INPUT)
        return
    finally:
        try:
            disconnect()
        except Exception:
            pass

    if not tickers:
        console.print(f"[yellow]Watchlist {args.name!r} is empty.[/yellow]")
        return

    console.print(f"Pulled {len(tickers)} ticker(s) from {args.name!r}: {', '.join(tickers)}")
    screen_args = argparse.Namespace(tickers=tickers)
    cmd_screen(screen_args)


def cmd_dashboard(args: argparse.Namespace) -> None:
    import webbrowser

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    console.print(f"[bold]stockoptions dashboard[/bold] starting at {url} (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run("stockoptions.web.app:app", host=args.host, port=args.port, log_level="warning")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stockoptions", description="Options analysis toolkit. For fun, not financial advice.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_greeks = sub.add_parser("greeks", help="price/Greeks for a full option chain, recomputed from real quotes")
    p_greeks.add_argument("ticker")
    p_greeks.add_argument("--expiration", required=True, help="YYYY-MM-DD")
    p_greeks.add_argument("--type", choices=["call", "put"], default="call")
    p_greeks.set_defaults(func=cmd_greeks)

    p_screen = sub.add_parser("screen", help="IV vs. realized-vol screen for one or more tickers")
    p_screen.add_argument("tickers", nargs="+")
    p_screen.set_defaults(func=cmd_screen)

    p_backtest = sub.add_parser("backtest", help="backtest the directional signal against honest baselines")
    p_backtest.add_argument("ticker")
    p_backtest.add_argument("--horizon", type=int, default=5, help="forward-looking days for the up/down label")
    p_backtest.add_argument("--period", default="2y", help="yfinance history period, e.g. 1y/2y/5y")
    p_backtest.set_defaults(func=cmd_backtest)

    p_strategy = sub.add_parser("strategy", help="max profit/loss/breakevens for a multi-leg position -- pure math, no prediction")
    strategy_sub = p_strategy.add_subparsers(dest="kind", required=True)

    p_vert = strategy_sub.add_parser("vertical")
    p_vert.add_argument("--option-type", choices=["call", "put"], dest="option_type", required=True)
    p_vert.add_argument("--long-strike", type=float, required=True)
    p_vert.add_argument("--short-strike", type=float, required=True)
    p_vert.add_argument("--long-premium", type=float, required=True)
    p_vert.add_argument("--short-premium", type=float, required=True)
    p_vert.set_defaults(func=cmd_strategy, kind="vertical")

    p_strangle = strategy_sub.add_parser("strangle")
    p_strangle.add_argument("--call-strike", type=float, required=True)
    p_strangle.add_argument("--put-strike", type=float, required=True)
    p_strangle.add_argument("--call-premium", type=float, required=True)
    p_strangle.add_argument("--put-premium", type=float, required=True)
    p_strangle.add_argument("--long", action="store_true", help="long strangle instead of the default short")
    p_strangle.set_defaults(func=cmd_strategy, kind="strangle")

    p_condor = strategy_sub.add_parser("iron-condor")
    for name in ["put-long-strike", "put-short-strike", "call-short-strike", "call-long-strike",
                 "put-long-premium", "put-short-premium", "call-short-premium", "call-long-premium"]:
        p_condor.add_argument(f"--{name}", type=float, required=True, dest=name.replace("-", "_"))
    p_condor.set_defaults(func=cmd_strategy, kind="iron-condor")

    p_watchlist = sub.add_parser("watchlist", help="pull tickers from your Robinhood watchlist and screen them (needs .env credentials)")
    p_watchlist.add_argument("--name", default="My First List")
    p_watchlist.set_defaults(func=cmd_watchlist)

    p_dashboard = sub.add_parser("dashboard", help="launch the local web dashboard")
    p_dashboard.add_argument("--host", default="127.0.0.1")
    p_dashboard.add_argument("--port", type=int, default=8000)
    p_dashboard.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    p_dashboard.set_defaults(func=cmd_dashboard)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except TickerNotFoundError as exc:
        _fail(str(exc), EXIT_BAD_INPUT)
    except NoArbitrageViolation as exc:
        _fail(str(exc), EXIT_BAD_INPUT)
    except ValueError as exc:
        _fail(str(exc), EXIT_BAD_INPUT)
    except Exception as exc:  # network hiccups, yfinance errors, etc.
        _fail(f"{type(exc).__name__}: {exc}", EXIT_DATA_ERROR)


if __name__ == "__main__":
    main()
