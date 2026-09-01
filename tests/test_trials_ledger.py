"""Tests for the trials ledger (Phase 3.1, 3.2)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from algos.wfov import trials_ledger as tl


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    return tmp_path / "trials_ledger.db"


def _make_trial(**overrides) -> tl.Trial:
    base = dict(
        proposed_at=datetime.now(timezone.utc).isoformat(),
        layer="weights",
        description="test trial",
        accepted=True,
        rationale="unit test",
        source_file="test.json",
    )
    base.update(overrides)
    return tl.Trial(**base)


class TestInsertAndCount:
    def test_empty_count_is_zero(self, tmp_db):
        assert tl.get_cumulative_n(tmp_db) == 0

    def test_insert_increments(self, tmp_db):
        tl.insert_trial(_make_trial(), db_path=tmp_db)
        assert tl.get_cumulative_n(tmp_db) == 1
        tl.insert_trial(_make_trial(), db_path=tmp_db)
        assert tl.get_cumulative_n(tmp_db) == 2

    def test_cumulative_n_written_to_row(self, tmp_db):
        tl.insert_trial(_make_trial(description="t1"), db_path=tmp_db)
        tl.insert_trial(_make_trial(description="t2"), db_path=tmp_db)
        rows = list(tl.iter_trials(db_path=tmp_db))
        assert rows[0]["cumulative_n"] == 1
        assert rows[1]["cumulative_n"] == 2


class TestIter:
    def test_filter_by_layer(self, tmp_db):
        tl.insert_trial(_make_trial(layer="weights"), db_path=tmp_db)
        tl.insert_trial(_make_trial(layer="retrain"), db_path=tmp_db)
        tl.insert_trial(_make_trial(layer="weights"), db_path=tmp_db)
        weights = list(tl.iter_trials(layer="weights", db_path=tmp_db))
        assert len(weights) == 2

    def test_filter_by_accepted(self, tmp_db):
        tl.insert_trial(_make_trial(accepted=True), db_path=tmp_db)
        tl.insert_trial(_make_trial(accepted=False), db_path=tmp_db)
        accepted = list(tl.iter_trials(accepted=True, db_path=tmp_db))
        rejected = list(tl.iter_trials(accepted=False, db_path=tmp_db))
        assert len(accepted) == 1
        assert len(rejected) == 1


class TestDSRComputation:
    def test_dsr_decreases_as_trials_increase(self, tmp_db):
        # Same observed Sharpe; DSR should be lower with more trials.
        r1 = tl.compute_dsr_at_current_n(
            observed_sharpe=1.5, n_observations=1260, db_path=tmp_db,
        )
        # Insert 100 dummy trials.
        for _ in range(100):
            tl.insert_trial(_make_trial(), db_path=tmp_db)
        r2 = tl.compute_dsr_at_current_n(
            observed_sharpe=1.5, n_observations=1260, db_path=tmp_db,
        )
        assert r2["deflated_sharpe"] < r1["deflated_sharpe"]


class TestBackfill:
    def test_backfill_inserts_one_per_summary(self, tmp_path, tmp_db):
        results = tmp_path / "summaries"
        results.mkdir()
        for i in range(3):
            payload = {
                "metadata": {
                    "model_name": "gnb", "ticker": f"T{i}",
                    "iterations_successful": 100,
                    "timestamp": "2026-05-01T00:00:00",
                },
                "performance_metrics": {
                    "sharpe_ratio": {"mean": 1.0},
                    "skewness": {"mean": 0.0},
                    "kurtosis": {"mean": 3.0},
                },
            }
            (results / f"montec_gnb_T{i}_summary.json").write_text(json.dumps(payload))
        result = tl.backfill_from_wfov(results_dir=results, db_path=tmp_db)
        assert result["scanned"] == 3
        assert result["inserted"] == 3
        # Backfill inserts layer='backtest' trials, which are excluded
        # from the scoped count (per REVISION_POLICY.md amendment 2026-07-07).
        # Use raw DB count for backfill assertions.
        with tl.open_db(tmp_db) as conn:
            raw_count = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        assert raw_count == 3

    def test_backfill_is_idempotent(self, tmp_path, tmp_db):
        results = tmp_path / "summaries"
        results.mkdir()
        (results / "montec_gnb_BK_summary.json").write_text(json.dumps({
            "metadata": {"model_name": "gnb", "ticker": "AAA",
                         "iterations_successful": 50},
            "performance_metrics": {
                "sharpe_ratio": {"mean": 1.0},
                "skewness": {"mean": 0.0},
                "kurtosis": {"mean": 3.0},
            },
        }))
        tl.backfill_from_wfov(results_dir=results, db_path=tmp_db)
        tl.backfill_from_wfov(results_dir=results, db_path=tmp_db)
        # Backfill inserts layer='backtest'; use raw DB count (scoped count
        # excludes backtest per 2026-07-07 amendment).
        with tl.open_db(tmp_db) as conn:
            raw_count = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        assert raw_count == 1  # not 2

    def test_backfill_dry_run_does_not_insert(self, tmp_path, tmp_db):
        results = tmp_path / "summaries"
        results.mkdir()
        (results / "montec_gnb_X_summary.json").write_text(json.dumps({
            "metadata": {"model_name": "gnb", "ticker": "X",
                         "iterations_successful": 10},
            "performance_metrics": {"sharpe_ratio": {"mean": 0.5}},
        }))
        result = tl.backfill_from_wfov(
            results_dir=results, db_path=tmp_db, dry_run=True
        )
        assert result["inserted"] == 1
        # But DB has nothing.
        assert tl.get_cumulative_n(tmp_db) == 0
