"""Test that trials_ledger scoped count excludes layer='backtest' trials.

Per REVISION_POLICY.md amendment 2026-07-07: the 50-trial retirement cap
and the DSR haircut both count only trials with layer IN
('weights','universe','retrain','architecture'), excluding layer='backtest'
legacy pre-protocol single-ticker R&D rows.
"""
import sqlite3
import tempfile
from pathlib import Path

from algos.wfov.trials_ledger import (
    Trial,
    compute_dsr_at_current_n,
    get_cumulative_n,
    insert_trial,
    open_db,
)


def _make_test_db(tmp_path: Path) -> Path:
    """Create a temp trials_ledger with 3 backtest + 2 weights trials."""
    db = tmp_path / "test_trials.db"
    with open_db(db) as conn:
        for i in range(3):
            conn.execute(
                "INSERT INTO trials (proposed_at, layer, description, cumulative_n, accepted) "
                "VALUES (?, 'backtest', ?, ?, 1)",
                (f"2026-03-28T00:00:0{i}", f"backtest trial {i}", i + 1),
            )
        for i in range(2):
            conn.execute(
                "INSERT INTO trials (proposed_at, layer, description, cumulative_n, accepted) "
                "VALUES (?, 'weights', ?, ?, 0)",
                (f"2026-05-17T00:00:0{i}", f"weights trial {i}", 4 + i),
            )
    return db


def test_get_cumulative_n_excludes_backtest_layer(tmp_path):
    """Scoped count should be 2 (weights only), not 5 (all layers)."""
    db = _make_test_db(tmp_path)
    scoped = get_cumulative_n(db)
    assert scoped == 2, f"Expected scoped count 2, got {scoped}"


def test_get_cumulative_n_all_includes_backtest(tmp_path):
    """Full count (for audit) should still be 5."""
    db = _make_test_db(tmp_path)
    with open_db(db) as conn:
        total = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    assert total == 5


def test_compute_dsr_uses_scoped_n(tmp_path):
    """DSR should use scoped N (3 = 2 existing + 1 proposed), not 6."""
    db = _make_test_db(tmp_path)
    result = compute_dsr_at_current_n(
        observed_sharpe=1.0,
        n_observations=252,
        db_path=db,
    )
    # n_trials in the result should be 3 (2 scoped + 1 proposed), not 6
    assert result["n_trials"] == 3, f"Expected n_trials=3 (scoped), got {result['n_trials']}"
