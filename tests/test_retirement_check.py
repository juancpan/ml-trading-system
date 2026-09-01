"""Tests for scripts/retirement_check.py (Phase 4.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import retirement_check as rc  # noqa: E402
from algos.wfov import trials_ledger


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "trials_ledger.db"
    monkeypatch.setattr(trials_ledger, "DEFAULT_DB", db)
    return db


class TestEvaluateRetirement:
    def test_clean_state_allows_continue(self, tmp_db):
        result = rc.evaluate_retirement()
        assert result["should_retire"] is False

    def test_trials_ceiling_triggers_retirement(self, tmp_db):
        for _ in range(rc._RETIREMENT_TRIALS_CEILING + 1):
            trials_ledger.insert_trial(
                trials_ledger.Trial(
                    proposed_at="2026-05-01T00:00:00",
                    layer="weights", description="dummy",
                    accepted=True,
                ),
                db_path=tmp_db,
            )
        result = rc.evaluate_retirement()
        assert result["should_retire"] is True
        assert any("ceiling" in f for f in result["findings"])

    def test_dsr_below_zero_triggers_retirement(self, tmp_db):
        # Inflate N to make DSR small.
        for _ in range(40):
            trials_ledger.insert_trial(
                trials_ledger.Trial(
                    proposed_at="2026-05-01T00:00:00",
                    layer="weights", description="x", accepted=True,
                ),
                db_path=tmp_db,
            )
        # Live Sharpe = 0.1 (very weak) over 252 obs = 1 year.
        result = rc.evaluate_retirement(
            live_sharpe=0.1, live_observations=252,
        )
        assert result["should_retire"] is True
        assert any("DSR" in f for f in result["findings"])

    def test_strong_live_sharpe_does_not_trigger(self, tmp_db):
        # Modest N, strong Sharpe → don't retire.
        result = rc.evaluate_retirement(
            live_sharpe=2.0, live_observations=504,
        )
        assert result["should_retire"] is False
