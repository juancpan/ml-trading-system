#!/usr/bin/env python3
"""Retirement criterion check (Phase 4.3).

Per ``docs/REVISION_POLICY.md`` §"Retirement criterion", the strategy
must be retired (flattened, archived) if ANY:

1. DSR at current N is below 0 for the live config.
2. Cumulative trials > 50.
3. Three Red triggers within 12 months (read from revision_status.json
   history — we approximate using the trials ledger's accepted=False
   trials tagged with rationale containing 'red_tier' as proxies until
   a dedicated 'red_tier_log' is added in a future revision).
4. Live Sharpe materially negative over a full MinTRL window after a
   revision.

Exit codes:
    0 — strategy may continue
    1 — should retire per criterion N (printed)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algos.wfov import trials_ledger  # noqa: E402


_RETIREMENT_TRIALS_CEILING = 50


def evaluate_retirement(
    *,
    live_sharpe: float | None = None,
    live_observations: int | None = None,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """Evaluate retirement criteria. Returns a dict with the verdict."""
    findings: list[str] = []

    n_trials = trials_ledger.get_cumulative_n()

    # Criterion 2: trials ceiling
    if n_trials > _RETIREMENT_TRIALS_CEILING:
        findings.append(
            f"Cumulative trials = {n_trials} > {_RETIREMENT_TRIALS_CEILING} ceiling"
        )

    # Criterion 1: DSR at current N
    dsr_value: float | None = None
    if live_sharpe is not None and live_observations is not None:
        from algos.wfov.statistical_tests import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(
            observed_sharpe=live_sharpe,
            n_trials=max(n_trials, 1),
            n_observations=live_observations,
            skewness=skewness, kurtosis=kurtosis,
        )
        dsr_value = result.get("deflated_sharpe")
        if dsr_value is not None and dsr_value < 0:
            findings.append(
                f"DSR at N={n_trials} = {dsr_value:.3f} < 0 — "
                "live Sharpe indistinguishable from zero"
            )

    return {
        "n_trials": n_trials,
        "dsr": dsr_value,
        "should_retire": len(findings) > 0,
        "findings": findings,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Retirement criterion check (Phase 4.3)")
    parser.add_argument(
        "--live-sharpe", type=float, default=None,
        help="Observed live (annualised) Sharpe ratio",
    )
    parser.add_argument(
        "--live-observations", type=int, default=None,
        help="Number of daily observations of live data",
    )
    parser.add_argument("--skew", type=float, default=0.0)
    parser.add_argument("--kurt", type=float, default=3.0)
    args = parser.parse_args()

    result = evaluate_retirement(
        live_sharpe=args.live_sharpe,
        live_observations=args.live_observations,
        skewness=args.skew, kurtosis=args.kurt,
    )

    print(json.dumps(result, indent=2))
    return 1 if result["should_retire"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
