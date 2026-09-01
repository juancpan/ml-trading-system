"""Tests for scripts/action_matrix.py (Phase 4.1)."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import action_matrix as am  # noqa: E402


class TestActionMatrix:
    def test_execution_drag_allowed_at_yellow(self):
        e = am.find_entry(attribution="execution_drag", current_tier="yellow")
        assert e is not None
        assert "tune_limit_price_strategy" in e.allowed_actions
        assert e.trials_cost == 0

    def test_signal_decay_major_requires_red(self):
        # Orange is not enough for the major-decay action.
        e_orange = am.find_entry(attribution="signal_decay_major", current_tier="orange")
        assert e_orange is None
        e_red = am.find_entry(attribution="signal_decay_major", current_tier="red")
        assert e_red is not None
        assert e_red.trials_cost == 5

    def test_unknown_attribution_returns_none(self):
        e = am.find_entry(attribution="bogus", current_tier="red")
        assert e is None

    def test_higher_tier_unlocks_lower_tier_actions(self):
        # weight_drift requires orange; red should also be allowed.
        e = am.find_entry(attribution="weight_drift", current_tier="red")
        assert e is not None

    def test_lower_tier_blocks_higher_tier_action(self):
        # weight_drift requires orange; yellow should NOT be allowed.
        e = am.find_entry(attribution="weight_drift", current_tier="yellow")
        assert e is None

    def test_list_attributions_covers_policy(self):
        attrs = am.list_attributions()
        for required in (
            "execution_drag", "weight_drift", "universe_regime_change",
            "signal_decay_minor", "signal_decay_major", "unattributable",
        ):
            assert required in attrs
