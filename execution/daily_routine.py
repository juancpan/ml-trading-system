#!/usr/bin/env python3
"""One-command daily routine driver (OPERATIONS_MANUAL §3).

READ-ONLY. Places no orders, modifies no state. It collapses the manual's
8-step daily checklist into a single command and, crucially, adds a
**missed-run / staleness detector** so a silent multi-day gap (like the
May 2026 outage) is surfaced immediately instead of being discovered weeks
later.

    python execution/daily_routine.py

Exit codes:
    0 = all green (ATTENTION items may still be printed as INFO)
    1 = one or more ATTENTION items (review before trusting the system)

This script intentionally has NO heavy imports at module load beyond stdlib
+ pandas (already a project dependency). It must run fast and never crash the
operator's morning check.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LOG_DIR = SCRIPT_DIR / "logs"

# Load .env so any credential-dependent sub-check behaves like cron does.
sys.path.insert(0, str(SCRIPT_DIR))
try:
    import env_loader  # noqa: F401
except Exception:
    pass

SENTINELS = ("KILL_SWITCH_ACTIVE", "SOFT_HALT_ACTIVE", "DAILY_MOVE_ACTIVE")

# Files whose freshness indicates the daily pipeline ran. (path, max_age_days)
FRESHNESS_TARGETS = (
    ("equity_history.parquet", 4),
    ("signal_history.parquet", 4),
    ("revision_status.json", 4),
)

# Regions and the weekdays (Mon=0..Sun=6) they are expected to run.
# Mirrors crontab_regions.txt.
REGION_DAYS = {
    "US": {0, 1, 2, 3, 4},
    "CANADA": {0, 1, 2, 3, 4},
    "EUROPE": {0, 1, 2, 3, 4},
   # "MIDDLE_EAST": {6, 0, 1, 2, 3},  # Sun-Thu
}

_GREEN = "OK"
_ATTN = "ATTENTION"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.attention = False

    def add(self, label: str, status: str, detail: str = "") -> None:
        if status == _ATTN:
            self.attention = True
        self.rows.append((label, status, detail))

    def render(self) -> str:
        width = max(len(r[0]) for r in self.rows)
        lines = []
        for label, status, detail in self.rows:
            mark = "✓" if status == _GREEN else "!"
            line = f"  [{mark}] {label.ljust(width)}  {status}"
            if detail:
                line += f" — {detail}"
            lines.append(line)
        return "\n".join(lines)


def _age_days(p: Path) -> float | None:
    if not p.exists():
        return None
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 86400.0


def regen_status(rep: Report) -> None:
    """Regenerate revision_status.json before reading it (best-effort).

    Nothing in the cron path writes revision_status.json — it is produced only
    by revision_triggers.evaluate(). Without this, the file stays frozen at
    whatever date it was last manually generated, and the Trigger/State checks
    below report a stale tier. We regenerate it here so the morning check
    reflects live data. This MUST NOT crash the routine: on any failure we
    log an ATTENTION line and fall through to the existing file.
    """
    try:
        import revision_triggers
        ev = revision_triggers.evaluate()
        revision_triggers.write_status(ev)
        rep.add("Status regen", _GREEN, f"revision_status.json refreshed (tier={ev.tier})")
    except Exception as exc:  # never block the routine
        rep.add("Status regen", _ATTN,
                f"regen failed: {type(exc).__name__}: {exc} — reading stale file")


def check_sentinels(rep: Report) -> None:
    present = [s for s in SENTINELS if (SCRIPT_DIR / s).exists()]
    if not present:
        rep.add("Sentinel scan", _GREEN, "no sentinels")
    else:
        rep.add("Sentinel scan", _ATTN, f"present: {', '.join(present)} (see §9.1)")


def check_last_cron(rep: Report) -> None:
    logs = sorted(LOG_DIR.glob("cron_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        rep.add("Last-cron health", _ATTN, "no cron logs found")
        return
    latest = logs[0]
    age = _age_days(latest)
    text = latest.read_text(encoding="utf-8", errors="replace")
    # Distinguish environment failure (exit 5 / Gate 0) from genuine preflight.
    if "ENVIRONMENT UNAVAILABLE" in text:
        rep.add("Last-cron health", _ATTN,
                f"{latest.name}: ENVIRONMENT failure (volume/FDA) — not a strategy issue")
    elif "Preflight FAILED" in text or "NAV READ FAILED" in text:
        rep.add("Last-cron health", _ATTN, f"{latest.name}: preflight/NAV failure")
    elif "Exit code: 0" in text or "Preflight PASSED" in text or "Gate 0 PASSED" in text:
        rep.add("Last-cron health", _GREEN, f"{latest.name} ({age:.1f}d old)")
    else:
        rep.add("Last-cron health", _ATTN, f"{latest.name}: indeterminate outcome")


def check_missed_runs(rep: Report) -> None:
    """The key anti-silent-gap detector: per region, is the newest log too old?"""
    today = datetime.now(timezone.utc)
    weekday = today.weekday()
    stale_regions = []
    for region, days in REGION_DAYS.items():
        logs = sorted(LOG_DIR.glob(f"cron_{region}_*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            stale_regions.append(f"{region}=NO LOGS")
            continue
        age = _age_days(logs[0])
        # On a trading weekday for the region, a >2-day-old newest log = missed runs.
        threshold = 2.0 if weekday in days else 4.0
        if age is not None and age > threshold:
            stale_regions.append(f"{region}={age:.0f}d")
    if stale_regions:
        rep.add("Missed-run detector", _ATTN,
                "stale regions: " + ", ".join(stale_regions) + " (pipeline may be down)")
    else:
        rep.add("Missed-run detector", _GREEN, "all regions ran recently")


def check_freshness(rep: Report) -> None:
    stale = []
    for name, max_age in FRESHNESS_TARGETS:
        age = _age_days(SCRIPT_DIR / name)
        if age is None:
            stale.append(f"{name}=MISSING")
        elif age > max_age:
            stale.append(f"{name}={age:.0f}d")
    if stale:
        rep.add("State freshness", _ATTN, ", ".join(stale))
    else:
        rep.add("State freshness", _GREEN, "equity/signal/status current")


def check_trigger_status(rep: Report) -> None:
    path = SCRIPT_DIR / "revision_status.json"
    if not path.exists():
        rep.add("Trigger status", _ATTN, "revision_status.json missing")
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        rep.add("Trigger status", _ATTN, f"unreadable: {e}")
        return
    tier = str(data.get("tier", "unknown")).lower()
    as_of = data.get("as_of", "?")
    age = _age_days(path)
    detail = f"tier={tier} as_of={as_of[:10]}"
    if age is not None and age > 4:
        detail += f" (STALE {age:.0f}d)"
    if tier in ("ok", "yellow") and (age is None or age <= 4):
        rep.add("Trigger status", _GREEN, detail)
    else:
        reasons = "; ".join(data.get("reasons", [])) or "see file"
        rep.add("Trigger status", _ATTN, f"{detail} — {reasons}")


def check_position_breakers(rep: Report) -> None:
    """Check for active per-position circuit breaker sentinels (Tier B)."""
    try:
        import position_circuit_breaker
        active = position_circuit_breaker.check_existing_breakers()
        if active:
            rep.add("Position breakers", _ATTN,
                    f"active: {', '.join(active)} (see §9 / position_circuit_breaker.py)")
        else:
            rep.add("Position breakers", _GREEN, "none active")
    except Exception:
        rep.add("Position breakers", _GREEN, "module not available")


def main() -> int:
    rep = Report()
    print("=" * 70)
    print(f"DAILY ROUTINE — {datetime.now().strftime('%Y-%m-%d %H:%M %Z')} (read-only)")
    print("OPERATIONS_MANUAL §3 — review ATTENTION items; this script never trades.")
    print("=" * 70)

    regen_status(rep)  # refresh revision_status.json before reading it
    check_sentinels(rep)
    check_last_cron(rep)
    check_missed_runs(rep)
    check_freshness(rep)
    check_trigger_status(rep)
    check_position_breakers(rep)

    print(rep.render())
    print("-" * 70)
    if rep.attention:
        print("RESULT: ATTENTION — one or more checks need your eyes. See §3 / §9.")
        print("Reminders not handled here (do manually): NAV snapshot, dashboard,")
        print("attribution sanity, per-model hit-rate, anomaly logging to")
        print("revision_hypotheses.md (observe today, act no sooner than MinTRL/4).")
        return 1
    print("RESULT: ALL GREEN. Still do the human-judgment steps (attribution,")
    print("hit-rate, anomaly logging) per §3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
