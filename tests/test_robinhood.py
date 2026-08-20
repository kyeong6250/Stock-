import ast
from pathlib import Path

import pytest

# Every order-placing/modifying/cancelling function robin_stocks exposes,
# plus the two watchlist *write* functions that live alongside the read
# ones in robin_stocks.robinhood.account. Sourced by listing
# robin_stocks.robinhood.orders' public functions and account's non-get_/
# non-build_ ones directly from the installed package.
_FORBIDDEN_NAMES = [
    "order_buy_crypto_by_price", "order_buy_crypto_by_quantity", "order_buy_crypto_limit",
    "order_buy_crypto_limit_by_price", "order_buy_fractional_by_price", "order_buy_fractional_by_quantity",
    "order_buy_limit", "order_buy_market", "order_buy_option_limit", "order_buy_option_stop_limit",
    "order_buy_stop_limit", "order_buy_stop_loss", "order_buy_trailing_stop", "order_crypto",
    "order_option_credit_spread", "order_option_debit_spread", "order_option_spread",
    "order_sell_crypto_by_price", "order_sell_crypto_by_quantity", "order_sell_crypto_limit",
    "order_sell_crypto_limit_by_price", "order_sell_fractional_by_price", "order_sell_fractional_by_quantity",
    "order_sell_limit", "order_sell_market", "order_sell_option_limit", "order_sell_option_stop_limit",
    "order_sell_stop_limit", "order_sell_stop_loss", "order_sell_trailing_stop", "order_trailing_stop",
    "cancel_all_crypto_orders", "cancel_all_option_orders", "cancel_all_stock_orders",
    "cancel_crypto_order", "cancel_option_order", "cancel_stock_order",
    "post_symbols_to_watchlist", "delete_symbols_from_watchlist",
]


def test_forbidden_trading_and_watchlist_write_functions_are_confirmed_real():
    """Sanity-check the blocklist itself: every name in it must actually
    exist somewhere in the installed robin_stocks package, or this test
    would be silently checking against typos instead of real function
    names."""
    import robin_stocks.robinhood as r

    all_names = set(dir(r.orders)) | set(dir(r.account))
    missing = [n for n in _FORBIDDEN_NAMES if n not in all_names]
    assert not missing, f"these aren't real robin_stocks function names: {missing}"


def test_robinhood_module_source_never_calls_or_imports_a_trading_or_write_function():
    """The actual, honest safety guarantee: none of robin_stocks' order-
    placing, -cancelling, or watchlist-modifying function names are ever
    imported or called in this file, so it has no way to invoke them --
    regardless of what robin_stocks' own __init__.py happens to import
    under the hood when the package loads. AST-based (not a substring
    search) specifically so mentioning these names in a comment/docstring
    -- like the one right above this function -- doesn't trip a false
    positive; only real imports and call expressions count."""
    tree = ast.parse(Path("stockoptions/robinhood.py").read_text())

    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            referenced.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                referenced.add(func.id)
            elif isinstance(func, ast.Attribute):
                referenced.add(func.attr)

    found = referenced & set(_FORBIDDEN_NAMES)
    assert not found, f"robinhood.py imports or calls forbidden function(s): {found}"


def test_load_credentials_raises_clear_error_when_unset(monkeypatch):
    from stockoptions.robinhood import RobinhoodAuthError, _load_credentials

    monkeypatch.delenv("ROBINHOOD_USERNAME", raising=False)
    monkeypatch.delenv("ROBINHOOD_PASSWORD", raising=False)
    with pytest.raises(RobinhoodAuthError):
        _load_credentials()


def test_load_credentials_reads_from_environment(monkeypatch):
    from stockoptions.robinhood import _load_credentials

    monkeypatch.setenv("ROBINHOOD_USERNAME", "someone@example.com")
    monkeypatch.setenv("ROBINHOOD_PASSWORD", "hunter2")
    monkeypatch.delenv("ROBINHOOD_MFA_CODE", raising=False)
    username, password, mfa = _load_credentials()
    assert username == "someone@example.com"
    assert password == "hunter2"
    assert mfa is None
