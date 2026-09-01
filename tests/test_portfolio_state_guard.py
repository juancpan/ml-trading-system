"""Guard: a non-live PortfolioManager must NOT clobber populated state files.

Regression for 2026-06-09: a test/demo PortfolioManager with empty
account_values overwrote the live account_values.pkl with {}, which made
nav_quick.py return 0 and the MIDDLE_EAST cron abort at the NAV gate.

See MEMORY.md "account_values.pkl clobbered to empty".
"""

from __future__ import annotations

import logging
import pickle

import pytest

from portfolio_manager import PortfolioManager


def _pm(tmp_path):
    pm = PortfolioManager(logger=logging.getLogger("guard_test"))
    # Redirect all state files into tmp so we never touch real ones.
    pm.account_state_file = tmp_path / "account_values.pkl"
    pm.positions_file = tmp_path / "positions.pkl"
    pm.json_state_file = tmp_path / "account_state.json"
    return pm


def test_empty_state_does_not_clobber_populated_account_values(tmp_path):
    """Empty in-memory account_values must not overwrite a populated file."""
    # Seed a populated live cache.
    live = {"NetLiquidation": {"value": 10700.22, "currency": "USD"}}
    with open(tmp_path / "account_values.pkl", "wb") as f:
        pickle.dump(live, f)

    pm = _pm(tmp_path)  # fresh PM => empty account_values
    assert pm.account_values == {}
    pm._save_state_for_oversight()

    # The populated file must survive untouched.
    with open(tmp_path / "account_values.pkl", "rb") as f:
        after = pickle.load(f)
    assert after == live
    assert after["NetLiquidation"]["value"] == 10700.22


def test_populated_state_is_written(tmp_path):
    """A live PM with real account_values writes normally."""
    pm = _pm(tmp_path)
    pm.account_values = {"NetLiquidation": {"value": 12345.0, "currency": "USD"}}
    pm._save_state_for_oversight()
    with open(tmp_path / "account_values.pkl", "rb") as f:
        after = pickle.load(f)
    assert after["NetLiquidation"]["value"] == 12345.0


def test_empty_state_writes_when_no_file_exists(tmp_path):
    """First-ever save with empty state may create the file (nothing to lose)."""
    pm = _pm(tmp_path)
    pm._save_state_for_oversight()
    assert (tmp_path / "account_values.pkl").exists()


def test_empty_positions_does_not_clobber(tmp_path):
    """Empty in-memory positions must not overwrite a populated positions file."""
    live = {"XYZ.MI": {"position": 73, "averageCost": 1.0,
                       "unrealizedPNL": 0.0, "realizedPNL": 0.0,
                       "accountName": "X", "contract": None}}
    with open(tmp_path / "positions.pkl", "wb") as f:
        pickle.dump(live, f)

    pm = _pm(tmp_path)  # empty current_positions
    pm._save_state_for_oversight()

    with open(tmp_path / "positions.pkl", "rb") as f:
        after = pickle.load(f)
    assert "XYZ.MI" in after
