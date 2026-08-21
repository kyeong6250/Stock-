from unittest.mock import patch

import pytest

from stockoptions.backtest import BacktestResult
from stockoptions.cli import build_parser, main
from stockoptions.recommend import KellyEdge, RecommendationError, SelectedContract, TradeRecommendation
from stockoptions.scanner import TickerScanResult


def test_build_parser_does_not_raise():
    parser = build_parser()
    assert parser is not None


def test_version_flag_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert "stockoptions" in capsys.readouterr().out


def test_missing_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args([])
    assert exc_info.value.code == 2


def test_strategy_iron_condor_end_to_end_no_network(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stockoptions", "strategy", "iron-condor",
            "--put-long-strike", "90", "--put-short-strike", "95",
            "--call-short-strike", "105", "--call-long-strike", "110",
            "--put-long-premium", "1", "--put-short-premium", "2",
            "--call-short-premium", "2", "--call-long-premium", "1",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert "2.00" in out  # max profit == the net credit, hand-verified in test_strategies.py
    assert "-3.00" in out  # max loss


def test_strategy_vertical_rejects_missing_required_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["strategy", "vertical", "--option-type", "call"])
    assert exc_info.value.code == 2  # argparse's own missing-required-arg error


def test_main_reports_value_errors_as_a_clean_message_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stockoptions", "strategy", "iron-condor",
            "--put-long-strike", "100", "--put-short-strike", "90",  # invalid order -> ValueError
            "--call-short-strike", "105", "--call-long-strike", "110",
            "--put-long-premium", "1", "--put-short-premium", "2",
            "--call-short-premium", "2", "--call-long-premium", "1",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    err = capsys.readouterr().out  # rich console prints to stdout by default
    assert "Traceback" not in err
    assert "stockoptions: error" in err


def test_backtest_subcommand_walk_forward_flag_defaults_to_off():
    args = build_parser().parse_args(["backtest", "AAPL"])
    assert args.walk_forward is False
    assert args.folds == 5


def test_backtest_subcommand_parses_walk_forward_flag():
    args = build_parser().parse_args(["backtest", "AAPL", "--walk-forward", "--folds", "3"])
    assert args.walk_forward is True
    assert args.folds == 3


def test_predict_subcommand_parses_with_defaults():
    args = build_parser().parse_args(["predict", "AAPL"])
    assert args.ticker == "AAPL"
    assert args.account_size == 10_000.0
    assert args.risk_pct == 0.02
    assert args.horizon == 35
    assert args.delta == 0.35
    assert args.model == "logistic"


def _fake_recommendation():
    bt = BacktestResult(
        accuracy=0.55,
        majority_baseline_accuracy=0.50,
        n_train_samples=100,
        n_test_samples=40,
        strategy_total_return=0.1,
        buy_and_hold_total_return=0.05,
        strategy_sharpe=1.2,
        strategy_max_drawdown=-0.05,
        dates=["2026-01-01"],
        strategy_equity_curve=[1.1],
        buy_and_hold_equity_curve=[1.05],
        feature_importance={"rsi_14": 0.5, "macd_hist_norm": 0.5},
    )
    contract = SelectedContract(
        expiration="2026-09-25", dte=36, option_type="call", strike=325.0, premium=4.7, iv=0.245, delta=0.316, T=0.0986, r=0.037
    )
    edge = KellyEdge(sample_size=49, win_rate=0.33, avg_win=13.5, avg_loss=3.8, kelly_fraction=0.14, insufficient_sample=False)
    return TradeRecommendation(
        ticker="AAPL",
        price=311.30,
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
        cone=[],
        warnings=["some warning"],
    )


def test_predict_command_prints_the_recommendation_and_warnings(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["stockoptions", "predict", "AAPL"])
    with patch("stockoptions.cli.recommend_trade", return_value=_fake_recommendation()) as mock_fn:
        main()
    out = capsys.readouterr().out
    mock_fn.assert_called_once_with("AAPL", account_size=10_000.0, max_risk_pct=0.02, horizon_days=35, target_delta=0.35, model_type="logistic")
    assert "AAPL" in out
    assert "call" in out
    assert "some warning" in out


def test_predict_command_reports_recommendation_errors_cleanly(capsys):
    with patch("stockoptions.cli.recommend_trade", side_effect=RecommendationError("no usable contracts")):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(["predict", "ZZZZZZ"]).func(build_parser().parse_args(["predict", "ZZZZZZ"]))
    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    assert "Traceback" not in out
    assert "no usable contracts" in out


def test_scan_subcommand_defaults_to_no_explicit_tickers():
    args = build_parser().parse_args(["scan"])
    assert args.tickers == []
    assert args.horizon == 5
    assert args.model == "logistic"
    assert args.sort_by == "volume"


def test_scan_subcommand_parses_explicit_tickers_and_options():
    args = build_parser().parse_args(["scan", "AAPL", "MSFT", "--sort-by", "iv", "--model", "random_forest"])
    assert args.tickers == ["AAPL", "MSFT"]
    assert args.sort_by == "iv"
    assert args.model == "random_forest"


def test_scan_command_uses_the_default_watchlist_when_no_tickers_given(monkeypatch, capsys):
    from stockoptions.scanner import DEFAULT_WATCHLIST

    monkeypatch.setattr("sys.argv", ["stockoptions", "scan"])
    fake_results = [TickerScanResult(ticker=t, price=100.0, volume_ratio=0.1, iv_hv_ratio=1.0, read="in line", direction="up", live_probability=0.55, backtest_accuracy=0.5, backtest_baseline=0.5, beats_baseline=False) for t in DEFAULT_WATCHLIST]
    with patch("stockoptions.cli.scan_tickers", return_value=fake_results) as mock_fn:
        main()
    mock_fn.assert_called_once_with(DEFAULT_WATCHLIST, horizon_days=5, model_type="logistic")
    out = capsys.readouterr().out
    assert DEFAULT_WATCHLIST[0] in out


def test_scan_command_reports_failed_tickers_separately_from_the_table(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["stockoptions", "scan", "AAPL", "BADTICKER"])
    fake_results = [
        TickerScanResult(ticker="AAPL", price=100.0, volume_ratio=0.1, iv_hv_ratio=1.0, read="in line", direction="up", live_probability=0.55, backtest_accuracy=0.5, backtest_baseline=0.5, beats_baseline=False),
        TickerScanResult(ticker="BADTICKER", error="no data found"),
    ]
    with patch("stockoptions.cli.scan_tickers", return_value=fake_results):
        main()
    out = capsys.readouterr().out
    assert "AAPL" in out
    assert "BADTICKER" in out
    assert "no data found" in out


def test_scan_command_sorts_by_the_requested_column(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["stockoptions", "scan", "LOW", "HIGH", "--sort-by", "accuracy"])
    fake_results = [
        TickerScanResult(ticker="LOW", price=100.0, volume_ratio=0.0, iv_hv_ratio=1.0, read="in line", direction="up", live_probability=0.55, backtest_accuracy=0.3, backtest_baseline=0.5, beats_baseline=False),
        TickerScanResult(ticker="HIGH", price=100.0, volume_ratio=0.0, iv_hv_ratio=1.0, read="in line", direction="up", live_probability=0.55, backtest_accuracy=0.7, backtest_baseline=0.5, beats_baseline=True),
    ]
    with patch("stockoptions.cli.scan_tickers", return_value=fake_results):
        main()
    out = capsys.readouterr().out
    assert out.index("HIGH") < out.index("LOW")  # higher accuracy sorts first
