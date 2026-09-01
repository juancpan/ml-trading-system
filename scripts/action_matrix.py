"""Action matrix (Phase 4.1).

Codifies the (attribution × tier) → allowed-action mapping from
``docs/REVISION_POLICY.md``. Consulted by ``scripts/run_revision.sh`` to
enforce the policy programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ActionEntry:
    attribution: str
    tier_required: str          # yellow | orange | red
    allowed_actions: tuple[str, ...]
    trials_cost: int            # for human reference; actual cost is N+1
    required_validation: tuple[str, ...]


# Same content as the table in docs/REVISION_POLICY.md.
ACTION_MATRIX: tuple[ActionEntry, ...] = (
    ActionEntry(
        attribution="execution_drag",
        tier_required="yellow",
        allowed_actions=("tune_limit_price_strategy", "tune_slippage_params"),
        trials_cost=0,
        required_validation=("ab_test_drag_week_over_week",),
    ),
    ActionEntry(
        attribution="weight_drift",
        tier_required="orange",
        allowed_actions=("rerun_hrp", "deploy_new_weights_json"),
        trials_cost=1,
        required_validation=("preflight", "shadow_one_week"),
    ),
    ActionEntry(
        attribution="universe_regime_change",
        tier_required="orange",
        allowed_actions=("drop_tickers_by_predeclared_filter",),
        trials_cost=1,
        required_validation=("preflight",),
    ),
    ActionEntry(
        attribution="signal_decay_minor",
        tier_required="orange",
        allowed_actions=("retrain_models_same_arch_same_features",),
        trials_cost=2,
        required_validation=("wfov", "dsr_with_inflated_n", "pbo"),
    ),
    ActionEntry(
        attribution="signal_decay_major",
        tier_required="red",
        allowed_actions=(
            "freeze_at_reduced_allocation",
            "architecture_review",
            "ensemble_reweighting",
        ),
        trials_cost=5,
        required_validation=("wfov", "dsr_with_inflated_n", "pbo", "newey_west"),
    ),
    ActionEntry(
        attribution="unattributable",
        tier_required="red",
        allowed_actions=("halt", "investigate", "consider_retirement"),
        trials_cost=0,
        required_validation=(),
    ),
)


def _tier_order(tier: str) -> int:
    return {"ok": 0, "yellow": 1, "orange": 2, "red": 3}.get(tier.lower(), 0)


def find_entry(*, attribution: str, current_tier: str) -> Optional[ActionEntry]:
    """Return the matching ActionEntry if (attribution, current_tier) is
    allowed, else None.

    The current_tier must be >= the entry's required tier in the
    OK<YELLOW<ORANGE<RED ordering.
    """
    cur = _tier_order(current_tier)
    for e in ACTION_MATRIX:
        if e.attribution != attribution:
            continue
        if cur >= _tier_order(e.tier_required):
            return e
    return None


def list_attributions() -> tuple[str, ...]:
    return tuple(e.attribution for e in ACTION_MATRIX)


def list_actions_for(attribution: str) -> tuple[str, ...]:
    for e in ACTION_MATRIX:
        if e.attribution == attribution:
            return e.allowed_actions
    return ()
