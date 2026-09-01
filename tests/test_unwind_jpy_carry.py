import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IBKR_DIR = PROJECT_ROOT / "execution"
sys.path.insert(0, str(IBKR_DIR))

from unwind_jpy_carry import (
    attach_whatif_error,
    choose_deleveraging_action,
    compute_unwind_sizing,
    whatif_preview_is_complete,
)


def test_compute_unwind_sizing_targets_corrected_leverage_ceiling():
    sizing = compute_unwind_sizing(
        nav=11209.59,
        leverage=1.3,
        jpy_balance=-3613973.0,
        usdjpy_rate=162.27,
    )

    assert math.isclose(sizing["ceiling_usd"], 3362.877, rel_tol=1e-6)
    assert math.isclose(sizing["jpy_debt_usd"], 22271.35638133974, rel_tol=1e-9)
    assert math.isclose(sizing["excess_usd"], 18908.47938133974, rel_tol=1e-9)
    assert math.isclose(sizing["post_unwind_jpy"], -545694.05079, rel_tol=1e-9)


def test_choose_deleveraging_action_selects_only_direction_that_reduces_margin():
    sell_preview = {"initMarginChange": -1200.0, "maintMarginChange": -1150.0}
    buy_preview = {"initMarginChange": 1200.0, "maintMarginChange": 1150.0}

    assert choose_deleveraging_action(sell_preview, buy_preview) == "SELL"


def test_choose_deleveraging_action_rejects_ambiguous_previews():
    sell_preview = {"initMarginChange": -100.0, "maintMarginChange": -100.0}
    buy_preview = {"initMarginChange": -50.0, "maintMarginChange": -50.0}

    assert choose_deleveraging_action(sell_preview, buy_preview) is None


def test_whatif_preview_waits_when_margin_fields_missing_despite_warning():
    preview = {
        "initMarginChange": None,
        "maintMarginChange": None,
        "warningText": "Important Note: order size is below IDEALPRO minimum",
    }

    assert not whatif_preview_is_complete(preview)


def test_whatif_preview_completes_when_order_specific_rejection_is_attached():
    preview = {"initMarginChange": None, "maintMarginChange": None}
    attach_whatif_error(
        preview,
        error_code=201,
        error_string="Order rejected - reason:Order increases leveraged FX position.",
    )

    assert whatif_preview_is_complete(preview)
    assert preview["errorCode"] == 201
    assert "increases leveraged FX position" in preview["errorString"]


def test_choose_deleveraging_action_uses_ibkr_leverage_rejection_when_margins_missing():
    sell_preview = {"initMarginChange": None, "maintMarginChange": None}
    buy_preview = {
        "initMarginChange": None,
        "maintMarginChange": None,
        "errorCode": 201,
        "errorString": "Order rejected - reason:Order increases leveraged FX position.",
    }

    assert choose_deleveraging_action(sell_preview, buy_preview) == "SELL"
