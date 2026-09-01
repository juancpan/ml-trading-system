#!/usr/bin/env python3
"""revision_check — advisory DSR check for revision proposals (Tier A).

Per REVISION_POLICY.md amendment 2026-07-07: DSR-gating is now advisory
(Tier A), not a hard block. This script computes and prints DSR/MinTRL
information but does NOT prevent the operator from proceeding. The
operator may override any Tier A gate by appending one written sentence
of rationale to the relevant docs/revision_hypotheses.md entry.

Reads a proposal YAML, computes DSR at the current cumulative trials
count, prints ADVISORY / WARN. If --commit is given, registers the trial
in the ledger.

Proposal YAML format:

    layer: weights | universe | retrain | architecture
    description: "Re-train AAA GNB on 2024-2026 data"
    ticker: AAA              # optional
    model_name: gnb         # optional
    source_wfov_run: algos/wfov/results/summaries/montec_gnb_BK_...json
    rationale: "Live hit-rate dropped to 0.43 over last 20 days"

Exit codes:
    0 — DSR computed and printed (advisory; operator decides)
    2 — Error (cannot read proposal / summary)

NOTE: exit code 1 (REJECT) is no longer used. The script always exits 0
if DSR was successfully computed, regardless of the DSR value. The DSR
value is printed as information, not as a gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algos.wfov.trials_ledger import (  # noqa: E402
    DEFAULT_DB, Trial, compute_dsr_at_current_n, get_cumulative_n,
    insert_trial,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight YAML reader (avoids forcing PyYAML as a hard dep)
# ---------------------------------------------------------------------------


def _read_proposal(path: Path) -> dict:
    text = path.read_text()
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception:
        # Fallback: very small key:value parser (one level deep)
        out: dict = {}
        for line in text.splitlines():
            line = line.split("#", 1)[0].rstrip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip().strip('"\'')
        return out


def _read_summary(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _git_head_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        return ""


def evaluate_proposal(proposal: dict) -> dict:
    """Compute DSR for the proposal and decide PASS/REJECT.

    Returns a dict with keys:
        decision: "pass" | "reject" | "error"
        dsr: float | None
        threshold: 0.5  (the required DSR floor — see docs/REVISION_POLICY.md)
        cumulative_n_before: int
        cumulative_n_after: int
        message: str
    """
    threshold = 0.5  # DSR threshold from REVISION_POLICY.md

    source = proposal.get("source_wfov_run")
    sharpe = proposal.get("observed_sharpe")
    n_obs = proposal.get("n_observations")
    skew = proposal.get("skewness", 0.0)
    kurt = proposal.get("kurtosis", 3.0)

    if source:
        sp = Path(source)
        if not sp.is_absolute():
            sp = REPO_ROOT / sp
        if not sp.exists():
            return {"decision": "error", "message": f"source_wfov_run not found: {sp}"}
        try:
            summary = _read_summary(sp)
            pm = summary.get("performance_metrics", {})
            sharpe = sharpe or (pm.get("sharpe_ratio", {}) or {}).get("mean")
            skew = (pm.get("skewness", {}) or {}).get("mean", 0.0)
            kurt = (pm.get("kurtosis", {}) or {}).get("mean", 3.0)
            meta = summary.get("metadata", {})
            n_obs = n_obs or meta.get("iterations_successful") or meta.get("iterations_requested")
        except Exception as exc:
            return {"decision": "error", "message": f"could not parse summary: {exc}"}

    if sharpe is None or n_obs is None:
        return {"decision": "error",
                "message": "missing observed_sharpe and/or n_observations"}

    n_before = get_cumulative_n()
    dsr_result = compute_dsr_at_current_n(
        observed_sharpe=float(sharpe),
        n_observations=int(n_obs),
        skewness=float(skew),
        kurtosis=float(kurt),
    )
    dsr = dsr_result.get("deflated_sharpe")
    n_after = n_before + 1
    pass_ = dsr is not None and not (isinstance(dsr, float) and dsr != dsr) and dsr >= threshold
    return {
        "decision": "pass" if pass_ else "reject",
        "dsr": dsr,
        "threshold": threshold,
        "cumulative_n_before": n_before,
        "cumulative_n_after": n_after,
        "observed_sharpe": float(sharpe),
        "n_observations": int(n_obs),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "interpretation": dsr_result.get("interpretation"),
        "message": "DSR-passed" if pass_ else "DSR below threshold at inflated N",
    }


def register(proposal: dict, eval_result: dict) -> int:
    """Register the trial in the ledger. Returns trial id."""
    trial = Trial(
        proposed_at=datetime.now(timezone.utc).isoformat(),
        layer=str(proposal.get("layer", "unknown")),
        description=str(proposal.get("description", "")),
        ticker=proposal.get("ticker"),
        model_name=proposal.get("model_name"),
        observed_sharpe=eval_result.get("observed_sharpe"),
        n_observations=eval_result.get("n_observations"),
        skewness=eval_result.get("skewness"),
        kurtosis=eval_result.get("kurtosis"),
        dsr_pre=eval_result.get("dsr"),
        dsr_haircut_at_pre=None,
        pbo_pre=None,
        accepted=(eval_result.get("decision") == "pass"),
        rationale=str(proposal.get("rationale", "")),
        source_file=str(proposal.get("source_wfov_run", "")),
        commit_hash=_git_head_hash(),
    )
    return insert_trial(trial)


def main() -> int:
    parser = argparse.ArgumentParser(description="Trials-budget revision checker")
    parser.add_argument("proposal", type=Path, help="Path to revision_proposal.yaml")
    parser.add_argument("--commit", action="store_true",
                        help="Insert into ledger regardless of pass/reject")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not args.proposal.exists():
        print(f"Proposal not found: {args.proposal}", file=sys.stderr)
        return 2

    proposal = _read_proposal(args.proposal)
    result = evaluate_proposal(proposal)

    print(f"Advisory: {result['decision'].upper()}")
    print(f"  message            = {result.get('message')}")
    print(f"  cumulative N (before this) = {result.get('cumulative_n_before')}")
    print(f"  cumulative N (with this)   = {result.get('cumulative_n_after')}")
    print(f"  DSR                = {result.get('dsr')}")
    print(f"  threshold          = {result.get('threshold')}")
    print(f"  interpretation     = {result.get('interpretation')}")
    print(f"  NOTE: DSR is advisory (Tier A). Operator may override with a")
    print(f"  logged rationale in docs/revision_hypotheses.md.")

    if result["decision"] == "error":
        return 2

    if args.commit:
        trial_id = register(proposal, result)
        print(f"  ledger trial_id    = {trial_id}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
