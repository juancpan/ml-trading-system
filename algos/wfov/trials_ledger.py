"""Trials Budget Ledger (Phase 3 of the Revision Protocol).

Every revision proposal — historical or new — is recorded as a row.
The Deflated Sharpe Ratio is recomputed at each proposal against the
INFLATED cumulative trial count. As N grows, the required DSR to claim
significance also grows. This is the price of multiple testing.

Schema::

    trials(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      proposed_at TEXT NOT NULL,     -- ISO timestamp
      layer TEXT NOT NULL,           -- weights | universe | retrain | architecture | backtest
      description TEXT NOT NULL,
      ticker TEXT,                   -- nullable; portfolio-wide trials have NULL
      model_name TEXT,               -- nullable
      observed_sharpe REAL,
      n_observations INTEGER,
      skewness REAL,
      kurtosis REAL,
      dsr_pre REAL,                  -- DSR at the cumulative N just BEFORE this trial
      dsr_haircut_at_pre REAL,       -- the haircut applied (multiple-testing penalty)
      pbo_pre REAL,                  -- PBO if known
      cumulative_n INTEGER,          -- N including this trial
      accepted INTEGER NOT NULL,     -- 0/1
      rationale TEXT,                -- human notes
      source_file TEXT,              -- WFOV summary path or similar
      commit_hash TEXT
    )
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "algos" / "wfov" / "trials_ledger.db"
DEFAULT_WFOV_RESULTS = REPO_ROOT / "algos" / "wfov" / "results" / "summaries"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_at TEXT NOT NULL,
    layer TEXT NOT NULL,
    description TEXT NOT NULL,
    ticker TEXT,
    model_name TEXT,
    observed_sharpe REAL,
    n_observations INTEGER,
    skewness REAL,
    kurtosis REAL,
    dsr_pre REAL,
    dsr_haircut_at_pre REAL,
    pbo_pre REAL,
    cumulative_n INTEGER NOT NULL,
    accepted INTEGER NOT NULL,
    rationale TEXT,
    source_file TEXT,
    commit_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_trials_layer ON trials(layer);
CREATE INDEX IF NOT EXISTS idx_trials_accepted ON trials(accepted);
CREATE INDEX IF NOT EXISTS idx_trials_source_file ON trials(source_file);
"""


@dataclass
class Trial:
    proposed_at: str
    layer: str
    description: str
    ticker: Optional[str] = None
    model_name: Optional[str] = None
    observed_sharpe: Optional[float] = None
    n_observations: Optional[int] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    dsr_pre: Optional[float] = None
    dsr_haircut_at_pre: Optional[float] = None
    pbo_pre: Optional[float] = None
    accepted: bool = True
    rationale: str = ""
    source_file: str = ""
    commit_hash: str = ""

    def to_row(self) -> tuple:
        return (
            self.proposed_at,
            self.layer,
            self.description,
            self.ticker,
            self.model_name,
            self.observed_sharpe,
            self.n_observations,
            self.skewness,
            self.kurtosis,
            self.dsr_pre,
            self.dsr_haircut_at_pre,
            self.pbo_pre,
            self.accepted,
            self.rationale,
            self.source_file,
            self.commit_hash,
        )


def _resolve_path(path: Optional[Path]) -> Path:
    """Resolve a db_path argument, using the module-level DEFAULT_DB if None.

    Reading the module-level attribute at call time (not import time)
    means monkey-patches in tests work as expected.
    """
    if path is None:
        # Read via module namespace so monkey-patches apply.
        import sys
        mod = sys.modules[__name__]
        return getattr(mod, "DEFAULT_DB")
    return path


@contextmanager
def open_db(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = _resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA_SQL)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_cumulative_n(db_path: Optional[Path] = None) -> int:
    """Scoped trial count for retirement-cap and DSR purposes.

    Counts only trials with ``layer IN ('weights','universe','retrain',
    'architecture')``, excluding ``layer='backtest'`` legacy pre-protocol
    single-ticker R&D rows. Per REVISION_POLICY.md amendment 2026-07-07.
    """
    db_path = _resolve_path(db_path)
    if not db_path.exists():
        return 0
    with open_db(db_path) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM trials WHERE layer != 'backtest'"
        )
        return int(cur.fetchone()[0])


def insert_trial(trial: Trial, *, db_path: Optional[Path] = None) -> int:
    """Insert a trial. Returns the assigned id."""
    with open_db(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO trials
                (proposed_at, layer, description, ticker, model_name,
                 observed_sharpe, n_observations, skewness, kurtosis,
                 dsr_pre, dsr_haircut_at_pre, pbo_pre,
                 cumulative_n, accepted, rationale, source_file, commit_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    (SELECT COUNT(*) + 1 FROM trials),
                    ?, ?, ?, ?)
            """,
            (
                trial.proposed_at, trial.layer, trial.description,
                trial.ticker, trial.model_name, trial.observed_sharpe,
                trial.n_observations, trial.skewness, trial.kurtosis,
                trial.dsr_pre, trial.dsr_haircut_at_pre, trial.pbo_pre,
                int(bool(trial.accepted)), trial.rationale,
                trial.source_file, trial.commit_hash,
            ),
        )
        return int(cur.lastrowid)


def iter_trials(
    *, layer: Optional[str] = None, accepted: Optional[bool] = None,
    db_path: Optional[Path] = None,
) -> Iterable[dict]:
    db_path = _resolve_path(db_path)
    if not db_path.exists():
        return iter([])
    sql = "SELECT * FROM trials WHERE 1=1"
    args: list = []
    if layer is not None:
        sql += " AND layer = ?"
        args.append(layer)
    if accepted is not None:
        sql += " AND accepted = ?"
        args.append(int(bool(accepted)))
    sql += " ORDER BY id"

    def _iter():
        with open_db(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for r in conn.execute(sql, args):
                yield dict(r)
    return _iter()


def compute_dsr_at_current_n(
    *,
    observed_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    db_path: Optional[Path] = None,
    annual_periods: int = 252,
) -> dict:
    """Compute DSR for a proposal using the CURRENT cumulative trials count.

    Adds 1 to the current N for the "after this trial" perspective —
    consistent with what insert_trial will record.
    """
    from algos.wfov.statistical_tests import deflated_sharpe_ratio

    db_path = _resolve_path(db_path)
    n_current = get_cumulative_n(db_path)
    n_with_this = max(n_current + 1, 1)
    return deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        n_trials=n_with_this,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        annual_periods=annual_periods,
    )


# ---------------------------------------------------------------------------
# Retroactive backfill (Phase 3.2)
# ---------------------------------------------------------------------------


def already_backfilled(source_file: str, *, db_path: Optional[Path] = None) -> bool:
    db_path = _resolve_path(db_path)
    if not db_path.exists():
        return False
    with open_db(db_path) as conn:
        cur = conn.execute(
            "SELECT 1 FROM trials WHERE source_file = ? LIMIT 1",
            (source_file,),
        )
        return cur.fetchone() is not None


def backfill_from_wfov(
    *,
    results_dir: Path = DEFAULT_WFOV_RESULTS,
    db_path: Optional[Path] = None,
    dry_run: bool = False,
) -> dict:
    db_path = _resolve_path(db_path)
    """Walk WFOV summaries, register each as a 'backtest' trial.

    Idempotent on ``source_file``.

    Returns counters: {scanned, inserted, skipped}.
    """
    scanned = 0
    inserted = 0
    skipped = 0
    for p in sorted(results_dir.glob("*_summary.json")):
        scanned += 1
        if already_backfilled(p.name, db_path=db_path):
            skipped += 1
            continue
        try:
            summary = json.loads(p.read_text())
        except Exception as exc:
            LOGGER.warning("Could not parse %s: %s", p.name, exc)
            skipped += 1
            continue
        meta = summary.get("metadata", {})
        pm = summary.get("performance_metrics", {})
        sharpe = (pm.get("sharpe_ratio", {}) or {}).get("mean")
        skew = (pm.get("skewness", {}) or {}).get("mean")
        kurt = (pm.get("kurtosis", {}) or {}).get("mean")
        iters = meta.get("iterations_successful") or meta.get("iterations_requested")

        ticker = meta.get("ticker")
        model = meta.get("model_name")
        ts = meta.get("timestamp") or datetime.now(timezone.utc).isoformat()
        trial = Trial(
            proposed_at=str(ts),
            layer="backtest",
            description=f"WFOV {model} on {ticker} ({iters} iterations)",
            ticker=ticker,
            model_name=model,
            observed_sharpe=float(sharpe) if sharpe is not None else None,
            n_observations=int(iters) if iters else None,
            skewness=float(skew) if skew is not None else None,
            kurtosis=float(kurt) if kurt is not None else None,
            accepted=True,
            rationale="retroactive backfill from WFOV summary",
            source_file=p.name,
        )
        if dry_run:
            inserted += 1
        else:
            insert_trial(trial, db_path=db_path)
            inserted += 1

    return {"scanned": scanned, "inserted": inserted, "skipped": skipped}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Trials ledger CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("count", help="Print cumulative trial count")
    sub.add_parser("list", help="List all trials")

    bf = sub.add_parser("backfill", help="Backfill from WFOV summaries")
    bf.add_argument("--results-dir", type=Path, default=DEFAULT_WFOV_RESULTS)
    bf.add_argument("--dry-run", action="store_true")

    dsr = sub.add_parser("dsr", help="Compute DSR at current cumulative N")
    dsr.add_argument("--sharpe", type=float, required=True)
    dsr.add_argument("--n-obs", type=int, required=True)
    dsr.add_argument("--skew", type=float, default=0.0)
    dsr.add_argument("--kurt", type=float, default=3.0)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "count":
        scoped = get_cumulative_n()
        # Also show full count for audit-trail transparency.
        with open_db() as conn:
            full = conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        print(f"scoped (retirement/DSR): {scoped}")
        print(f"full (audit trail):      {full}")
        return 0
    if args.cmd == "list":
        for r in iter_trials():
            print(r)
        return 0
    if args.cmd == "backfill":
        result = backfill_from_wfov(
            results_dir=args.results_dir, dry_run=args.dry_run,
        )
        print(f"scanned={result['scanned']} inserted={result['inserted']} "
              f"skipped={result['skipped']}")
        return 0
    if args.cmd == "dsr":
        result = compute_dsr_at_current_n(
            observed_sharpe=args.sharpe,
            n_observations=args.n_obs,
            skewness=args.skew, kurtosis=args.kurt,
        )
        for k, v in result.items():
            print(f"  {k:>20}: {v}")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
