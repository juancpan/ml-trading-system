"""Tests for scripts/revision_check.py (Phase 3.3)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

# Add the scripts dir for direct import.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import revision_check  # type: ignore # noqa: E402
from algos.wfov import trials_ledger  # noqa: E402


def _make_summary(sharpe: float = 1.5, iters: int = 1000) -> dict:
    return {
        "metadata": {
            "model_name": "gnb", "ticker": "AAA",
            "iterations_successful": iters,
        },
        "performance_metrics": {
            "sharpe_ratio": {"mean": sharpe},
            "skewness": {"mean": 0.0},
            "kurtosis": {"mean": 3.0},
        },
    }


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "trials_ledger.db"
    monkeypatch.setattr(trials_ledger, "DEFAULT_DB", db)
    monkeypatch.setattr(revision_check, "DEFAULT_DB", db, raising=False)
    return db


class TestEvaluateProposal:
    def test_passes_with_strong_sharpe_and_low_n(self, tmp_path, tmp_db):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(_make_summary(sharpe=3.0, iters=1500)))
        proposal = {
            "layer": "retrain",
            "description": "test",
            "source_wfov_run": str(summary_path),
        }
        result = revision_check.evaluate_proposal(proposal)
        assert result["decision"] == "pass"
        assert result["dsr"] > 0.5

    def test_rejects_with_weak_sharpe(self, tmp_path, tmp_db):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(_make_summary(sharpe=0.3, iters=500)))
        proposal = {
            "layer": "retrain",
            "description": "weak",
            "source_wfov_run": str(summary_path),
        }
        result = revision_check.evaluate_proposal(proposal)
        assert result["decision"] == "reject"

    def test_rejects_after_inflated_n(self, tmp_path, tmp_db):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(_make_summary(sharpe=1.5, iters=1000)))

        # Insert many dummy trials to inflate N.
        for _ in range(200):
            trials_ledger.insert_trial(
                trials_ledger.Trial(
                    proposed_at="2026-05-01T00:00:00",
                    layer="weights", description="dummy", accepted=True,
                ),
                db_path=tmp_db,
            )

        proposal = {
            "layer": "retrain",
            "description": "after inflation",
            "source_wfov_run": str(summary_path),
        }
        result = revision_check.evaluate_proposal(proposal)
        # SR=1.5 with N=200 trials and 1000 obs — DSR will be lower than threshold.
        assert result["decision"] == "reject"

    def test_error_on_missing_summary(self, tmp_db):
        result = revision_check.evaluate_proposal({
            "layer": "retrain", "description": "x",
            "source_wfov_run": "/nonexistent/path.json",
        })
        assert result["decision"] == "error"

    def test_error_on_missing_sharpe(self, tmp_db):
        result = revision_check.evaluate_proposal({
            "layer": "retrain", "description": "x",
        })
        assert result["decision"] == "error"


class TestRegister:
    def test_register_inserts_trial(self, tmp_path, tmp_db):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(_make_summary()))
        proposal = {
            "layer": "retrain", "description": "wrote it",
            "source_wfov_run": str(summary_path),
        }
        result = revision_check.evaluate_proposal(proposal)
        n_before = trials_ledger.get_cumulative_n(tmp_db)
        trial_id = revision_check.register(proposal, result)
        assert trial_id > 0
        assert trials_ledger.get_cumulative_n(tmp_db) == n_before + 1
