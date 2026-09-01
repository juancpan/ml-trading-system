"""Tests for PortfolioManager symbol resolution + key reconciliation.

Guards the fix for the EUR->"SBF"/Paris inference bug that stored European
instruments under wrong ".PA" keys (e.g. XYZ.PA instead of XYZ.MI), causing
the same instrument to be tracked under two keys.

See MEMORY.md "European position stored under wrong exchange key (.PA)".
"""

from __future__ import annotations

import logging
import types

import pytest

from portfolio_manager import PortfolioManager


class _FakeContract:
    """Minimal stand-in for ibapi Contract."""

    def __init__(self, symbol, exchange="", currency="EUR"):
        self.symbol = symbol
        self.exchange = exchange
        self.currency = currency


class _FakeExchangeManager:
    """Stub: mimics the lossy reverse conversion the bug relied on.

    For a Paris-style exchange ("SBF") it would historically produce ".PA".
    The point of the fix is that configured tickers never reach this path.
    """

    def ibkr_to_yfinance_symbol(self, ibkr_symbol, exchange):
        suffix_by_exchange = {"SBF": ".PA", "BVME": ".MI"}
        return ibkr_symbol + suffix_by_exchange.get(exchange, "")


def _pm():
    logger = logging.getLogger("test_pm")
    pm = PortfolioManager(logger=logger, exchange_manager=_FakeExchangeManager())
    return pm


def _update(pm, contract, position=10.0):
    pm.update_position(
        contract=contract,
        position=position,
        marketPrice=1.0,
        marketValue=position,
        averageCost=1.0,
        unrealizedPNL=0.0,
        realizedPNL=0.0,
        accountName="TEST",
    )


# ---------------------------------------------------------------------------
# Resolution: a configured Milan ticker must ALWAYS land on .MI regardless of
# what exchange (empty / SMART / BVME) IBKR returns on the callback.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exchange", ["", "SMART", "BVME"])
def test_isp_resolves_to_mi_not_pa(exchange):
    """XYZ (a Milan-listed issuer) must store as XYZ.MI for any exchange value."""
    pm = _pm()
    # Skip if XYZ.MI isn't in the live config (keeps test honest to config).
    if "XYZ" not in pm._bare_to_config or pm._bare_to_config["XYZ"] != "XYZ.MI":
        pytest.skip("XYZ.MI not in current config; resolution map is config-driven")
    _update(pm, _FakeContract("XYZ", exchange=exchange, currency="EUR"))
    assert "XYZ.MI" in pm.current_positions
    assert "XYZ.PA" not in pm.current_positions


def test_unconfigured_symbol_still_uses_fallback():
    """A symbol NOT in config falls through to exchange-manager conversion."""
    pm = _pm()
    _update(pm, _FakeContract("ZZZ", exchange="SBF", currency="EUR"))
    # Not configured -> reverse map returns None -> fallback yields ZZZ.PA
    assert "ZZZ.PA" in pm.current_positions


def test_us_ticker_unchanged():
    """US-style bare ticker (no suffix) stores under itself, not mangled."""
    pm = _pm()
    _update(pm, _FakeContract("AVGO", exchange="SMART", currency="USD"))
    assert "AVGO" in pm.current_positions


# ---------------------------------------------------------------------------
# Reconciler: existing stray keys collapse to canonical; idempotent.
# ---------------------------------------------------------------------------

def test_reconciler_drops_duplicate_when_canonical_exists():
    pm = _pm()
    if pm._bare_to_config.get("XYZ") != "XYZ.MI":
        pytest.skip("XYZ.MI not in current config")
    pm.current_positions["XYZ.MI"] = {"position": 73}
    pm.current_positions["XYZ.PA"] = {"position": 73}
    changes = pm.reconcile_position_keys()
    assert "XYZ.PA" not in pm.current_positions
    assert "XYZ.MI" in pm.current_positions
    assert any(c["action"] == "drop_stray_duplicate" for c in changes)


def test_reconciler_renames_when_only_stray_exists():
    pm = _pm()
    if pm._bare_to_config.get("XYZ") != "XYZ.MI":
        pytest.skip("XYZ.MI not in current config")
    pm.current_positions["XYZ.PA"] = {"position": 73}
    changes = pm.reconcile_position_keys()
    assert pm.current_positions.get("XYZ.MI", {}).get("position") == 73
    assert "XYZ.PA" not in pm.current_positions
    assert any(c["action"] == "rename_to_canonical" for c in changes)


def test_reconciler_is_idempotent():
    pm = _pm()
    if pm._bare_to_config.get("XYZ") != "XYZ.MI":
        pytest.skip("XYZ.MI not in current config")
    pm.current_positions["XYZ.PA"] = {"position": 73}
    pm.reconcile_position_keys()
    second = pm.reconcile_position_keys()
    assert second == []


def test_reconciler_leaves_clean_state_untouched():
    pm = _pm()
    pm.current_positions["AVGO"] = {"position": 5}
    pm.current_positions["BIL"] = {"position": 19}
    changes = pm.reconcile_position_keys()
    assert changes == []
    assert set(pm.current_positions) == {"AVGO", "BIL"}
