import pytest

from stockoptions.cli import build_parser, main


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
