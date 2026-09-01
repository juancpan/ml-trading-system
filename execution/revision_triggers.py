"""Revision-tier trigger logic (Phase 2.3 of the Revision Protocol).

Yellow / Orange / Red tiers, derived from the baseline distribution
percentiles produced by ``algos/wfov/baseline_distributions.py`` and the
live attribution / equity history.

Output: ``execution/revision_status.json``. Read by humans and the
``revision_dashboard``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, date as Date
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent

DEFAULT_STATUS = _THIS_DIR / "revision_status.json"
DEFAULT_EQUITY = _THIS_DIR / "equity_history.parquet"
DEFAULT_ATTR_DB = _THIS_DIR / "attribution.db"
DEFAULT_BASELINES_DIR = REPO_ROOT / "algos" / "wfov" / "baselines"


class RevisionTier(str, Enum):
    OK = "ok"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class TriggerEvaluation:
    as_of: str
    tier: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _load_latest_baseline(
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
) -> Optional[dict]:
    if not baselines_dir.exists():
        return None
    candidates = sorted(baselines_dir.glob("portfolio_*_baseline.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    try:
        return json.loads(candidates[0].read_text())
    except Exception as exc:
        LOGGER.error("Could not load baseline %s: %s", candidates[0], exc)
        return None


def _latest_drawdown_from_equity(equity_path: Path) -> Optional[float]:
    if not equity_path.exists():
        return None
    df = pd.read_parquet(equity_path)
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    # MTD drawdown across all regions, weighted by latest snapshot.
    latest = df.iloc[-1]
    as_of = latest["timestamp"]
    month_start = pd.Timestamp(
        datetime(as_of.year, as_of.month, 1, tzinfo=timezone.utc)
    )
    month_slice = df[df["timestamp"] >= month_start]
    if month_slice.empty:
        return None
    nav_open = float(month_slice["nav_usd"].iloc[0])
    nav_close = float(month_slice["nav_usd"].iloc[-1])
    return (nav_close / nav_open) - 1.0 if nav_open > 0 else None


def _recent_attribution(
    db_path: Path = DEFAULT_ATTR_DB, *, days: int = 60,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(db_path)) as conn:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM attribution_daily ORDER BY date DESC LIMIT ?",
                conn, params=(days,),
            )
        except Exception:
            return pd.DataFrame()
    return df


def _recent_hit_rates(db_path: Path = DEFAULT_ATTR_DB) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(db_path)) as conn:
        try:
            df = pd.read_sql_query(
                "SELECT * FROM model_hit_rate WHERE window_days = 20 "
                "ORDER BY date DESC LIMIT 200",
                conn,
            )
        except Exception:
            return pd.DataFrame()
    return df


def evaluate(
    *,
    equity_path: Path = DEFAULT_EQUITY,
    attr_db: Path = DEFAULT_ATTR_DB,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
    backtest_max_dd: Optional[float] = None,
) -> TriggerEvaluation:
    """Run the three-tier evaluation. Returns a TriggerEvaluation.

    Tier rules (ANY of the conditions triggers the tier):

      RED:
        - Live MTD drawdown < 1.2 × backtest 95th-percentile drawdown
        - All ml_signal models' 20d hit-rate < 0.5 simultaneously
      ORANGE:
        - MTD drawdown below backtest 5th percentile of monthly returns
        - Any model's 20d hit-rate < 0.45
        - (Execution-drag/other Phase 1.4-extended signals to be added)
      YELLOW:
        - MTD drawdown below backtest 25th percentile
      OK:
        - None of the above.
    """
    as_of = datetime.now(timezone.utc)
    reasons: list[str] = []
    metrics: dict = {"as_of": as_of.isoformat()}

    baseline = _load_latest_baseline(baselines_dir)
    bm = baseline.get("portfolio_aggregated_metrics", {}) if baseline else {}

    mtd_dd = _latest_drawdown_from_equity(equity_path)
    metrics["mtd_drawdown"] = mtd_dd

    p25_return = bm.get("annual_return.p25")
    p5_return = bm.get("annual_return.min")  # we don't have p5 directly; min is conservative
    p95_dd = bm.get("max_drawdown.p95")  # this is actually most negative dd in backtest
    metrics["backtest_p25_annual_return"] = p25_return
    metrics["backtest_p95_drawdown"] = p95_dd

    # Project annual return percentile thresholds onto monthly equivalents:
    # rough conversion: monthly_threshold ~ annual / 12. Crude but
    # acceptable for Phase 2; Phase 2.1 will refine with rolling 20d
    # percentiles when we have proper portfolio-level backtest output.
    p25_monthly = (p25_return / 12.0) if p25_return is not None else None
    p5_monthly = (p5_return / 12.0) if p5_return is not None else None

    # Backtest max drawdown — use the most conservative anchor.
    if backtest_max_dd is None and p95_dd is not None:
        # p95 is mean across tickers of (most negative DD * 0.95 percentile).
        # Apply a 1.2x safety factor for Red tier.
        backtest_max_dd = float(p95_dd)
    metrics["backtest_max_dd_anchor"] = backtest_max_dd

    hit_rates = _recent_hit_rates(attr_db)
    metrics["n_hit_rate_rows"] = int(len(hit_rates))

    # ----------------------------------------------------------
    # RED tier checks (strongest)
    # ----------------------------------------------------------
    tier = RevisionTier.OK

    if mtd_dd is not None and backtest_max_dd is not None:
        # backtest_max_dd is negative; threshold = 1.2 * backtest_max_dd
        threshold = 1.2 * backtest_max_dd
        if mtd_dd < threshold:
            tier = RevisionTier.RED
            reasons.append(
                f"MTD drawdown {mtd_dd:+.2%} below 1.2x backtest max DD ({threshold:+.2%})"
            )

    if not hit_rates.empty:
        latest = hit_rates.sort_values(
            ["ticker", "model_type", "date"], ascending=[True, True, False]
        ).drop_duplicates(subset=["ticker", "model_type"])
        if not latest.empty:
            latest_valid = latest.dropna(subset=["hit_rate"])
            metrics["n_ml_models"] = int(len(latest_valid))
            metrics["n_unresolved_ml_models"] = int(len(latest) - len(latest_valid))
            below_chance = (latest_valid["hit_rate"] < 0.5).sum()
            metrics["models_below_chance_20d"] = int(below_chance)
            if below_chance == len(latest_valid) and len(latest_valid) > 0:
                tier = RevisionTier.RED
                reasons.append(
                    "All ml_signal models below 50% hit-rate over 20d window"
                )

    # ----------------------------------------------------------
    # ORANGE tier — only if not already RED
    # ----------------------------------------------------------
    if tier != RevisionTier.RED:
        if mtd_dd is not None and p5_monthly is not None and mtd_dd < p5_monthly:
            tier = RevisionTier.ORANGE
            reasons.append(
                f"MTD return {mtd_dd:+.2%} below 5th percentile of backtest monthly returns ({p5_monthly:+.2%})"
            )
        if not hit_rates.empty:
            latest = hit_rates.sort_values(
                ["ticker", "model_type", "date"], ascending=[True, True, False]
            ).drop_duplicates(subset=["ticker", "model_type"])
            latest_valid = latest.dropna(subset=["hit_rate"])
            worst = float(latest_valid["hit_rate"].min()) if not latest_valid.empty else None
            metrics["worst_model_hit_rate_20d"] = worst
            if worst is not None and worst < 0.45:
                if tier == RevisionTier.OK or tier == RevisionTier.YELLOW:
                    tier = RevisionTier.ORANGE
                reasons.append(
                    f"At least one model hit-rate {worst:.2f} < 0.45 over 20d window"
                )

    # ----------------------------------------------------------
    # YELLOW tier — only if not Orange or Red
    # ----------------------------------------------------------
    if tier == RevisionTier.OK:
        if mtd_dd is not None and p25_monthly is not None and mtd_dd < p25_monthly:
            tier = RevisionTier.YELLOW
            reasons.append(
                f"MTD return {mtd_dd:+.2%} below 25th percentile of backtest monthly returns ({p25_monthly:+.2%})"
            )

    if not reasons:
        reasons.append("All monitored metrics within expected backtest ranges.")

    return TriggerEvaluation(
        as_of=as_of.isoformat(),
        tier=tier.value,
        reasons=reasons,
        metrics=metrics,
    )


def write_status(evaluation: TriggerEvaluation, *, path: Path = DEFAULT_STATUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": evaluation.as_of,
        "tier": evaluation.tier,
        "reasons": evaluation.reasons,
        "metrics": evaluation.metrics,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Revision-tier trigger evaluation (Phase 2.3)")
    parser.add_argument("--equity", type=Path, default=DEFAULT_EQUITY)
    parser.add_argument("--attr-db", type=Path, default=DEFAULT_ATTR_DB)
    parser.add_argument("--baselines-dir", type=Path, default=DEFAULT_BASELINES_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ev = evaluate(
        equity_path=args.equity,
        attr_db=args.attr_db,
        baselines_dir=args.baselines_dir,
    )
    write_status(ev, path=args.out)
    print(f"Tier: {ev.tier}")
    for r in ev.reasons:
        print(f"  - {r}")
    # Exit code for cron alerting
    return {"ok": 0, "yellow": 0, "orange": 1, "red": 2}[ev.tier]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
