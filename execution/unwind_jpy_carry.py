#!/usr/bin/env python3
"""One-off: unwind excess JPY carry debt to the corrected ceiling.

The carry-trade ceiling bug (fixed 2026-07-06 in cash_portfolio_manager.py)
allowed JPY debt to grow to ~$22k (~200% of NAV) when the intended ceiling
is (GENERAL_LEVERAGE - 1) * NAV ≈ $3.3k. This script:

  1. Connects to IBKR Gateway.
  2. Fetches live NAV, JPY cash balance, and USDJPY rate.
  3. Computes the corrected ceiling and the excess to unwind.
  4. Displays a summary.
  5. Optionally previews both SELL and BUY via IBKR what-if orders.
  6. Waits for typed "UNWIND" confirmation.
  7. Places the IBKR-validated de-risking order.
  8. Disconnects.

Usage:
    conda activate <your-env>
    python execution/unwind_jpy_carry.py --dry-run
    python execution/unwind_jpy_carry.py --whatif
    python execution/unwind_jpy_carry.py

Requires: IBKR Gateway running on 127.0.0.1:4002 (paper) or :4001 (live).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

# Bootstrap execution on sys.path so imports resolve when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import env_loader  # noqa: F401  (side-effect: populates os.environ from .env)
except Exception:
    pass

from ibapi.contract import Contract
from ibapi.order import Order

from ib_client_final import IBClient
from config import GENERAL_LEVERAGE, IB_HOST, IB_PORT, IB_CLIENT_ID


# ---------------------------------------------------------------------------
# Minimal stubs — IBClient's callbacks touch a few portfolio_manager /
# order_manager methods. We only need account_values and currency_balances
# for this one-off, so we stub the rest rather than dragging in the full
# PortfolioManager / OrderManager stack.
# ---------------------------------------------------------------------------


class StubPortfolioManager:
    """Minimal portfolio_manager for IBClient callbacks (account values only)."""

    def __init__(self, logger):
        self.logger = logger
        self.account_values = {}
        self.current_positions = {}
        self.initial_position_load_mode = False

    def update_account_value(self, key, value, currency, accountName):
        self.account_values[key] = {
            "value": value,
            "currency": currency,
            "accountName": accountName,
        }

    def update_position(self, *args, **kwargs):
        pass  # Not needed for unwind (position callbacks from reqAccountUpdates)

    def reconcile_position_keys(self, dry_run=True):
        return []  # No-op

    def get_current_net_liquidation(self):
        val = self.account_values.get("NetLiquidation", {}).get("value")
        try:
            return float(val) if val is not None else 0.0
        except (TypeError, ValueError):
            return 0.0


class StubOrderManager:
    """Minimal order_manager for IBClient callbacks."""

    def __init__(self):
        self.next_order_id = -1
        self._open = {}

    def set_next_order_id(self, oid):
        self.next_order_id = oid

    def get_open_orders(self):
        return self._open

    def track_order(self, oid, contract, order):
        self._open[oid] = {"contract": contract, "order": order}

    def update_order_status(self, oid, status, *args, **kwargs):
        pass

    def get_order_info(self, oid):
        return self._open.get(oid)


class WhatIfIBClient(IBClient):
    """IBClient variant that captures what-if OrderState payloads."""

    def __init__(self, portfolio_manager, order_manager, logger, client_id=None):
        super().__init__(portfolio_manager, order_manager, logger, client_id=client_id)
        self.whatif_events = {}
        self.whatif_states = {}

    def openOrder(self, orderId, contract, order, orderState):
        super().openOrder(orderId, contract, order, orderState)
        if getattr(order, "whatIf", False):
            self.whatif_states[orderId] = serialize_order_state(orderState)
            event = self.whatif_events.get(orderId)
            if event and whatif_preview_is_complete(self.whatif_states[orderId]):
                event.set()

    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
        super().error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)
        if reqId in self.whatif_events:
            state = self.whatif_states.setdefault(reqId, {})
            attach_whatif_error(state, error_code=errorCode, error_string=errorString)
            self.whatif_events[reqId].set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def setup_logger():
    logger = logging.getLogger("unwind_jpy")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def _to_float(value):
    if value in (None, "", "1.7976931348623157E308"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def serialize_order_state(order_state):
    """Extract margin-impact fields from IBKR OrderState."""
    numeric_fields = [
        "initMarginBefore",
        "initMarginChange",
        "initMarginAfter",
        "maintMarginBefore",
        "maintMarginChange",
        "maintMarginAfter",
        "equityWithLoanBefore",
        "equityWithLoanChange",
        "equityWithLoanAfter",
        "commissionAndFees",
        "minCommissionAndFees",
        "maxCommissionAndFees",
    ]
    state = {field: _to_float(getattr(order_state, field, None)) for field in numeric_fields}
    state["marginCurrency"] = getattr(order_state, "marginCurrency", "")
    state["status"] = getattr(order_state, "status", "")
    state["warningText"] = getattr(order_state, "warningText", "")
    state["rejectReason"] = getattr(order_state, "rejectReason", "")
    return state


def attach_whatif_error(preview, *, error_code, error_string):
    """Attach an order-specific IBKR error to a what-if preview dict."""
    preview["errorCode"] = int(error_code)
    preview["errorString"] = str(error_string)
    return preview


def whatif_preview_is_complete(preview):
    """A preview is terminal once margin fields arrive or IBKR rejects it."""
    if preview.get("errorCode") is not None or preview.get("rejectReason"):
        return True
    return (
        preview.get("initMarginChange") is not None
        and preview.get("maintMarginChange") is not None
    )


def compute_unwind_sizing(nav, leverage, jpy_balance, usdjpy_rate):
    """Compute the excess JPY debt above the leverage-derived ceiling."""
    ceiling_usd = max(0.0, (leverage - 1.0) * nav)
    jpy_debt_usd = abs(jpy_balance / usdjpy_rate) if jpy_balance < 0 else 0.0
    excess_usd = max(0.0, jpy_debt_usd - ceiling_usd)
    excess_jpy = excess_usd * usdjpy_rate
    return {
        "ceiling_usd": ceiling_usd,
        "jpy_debt_usd": jpy_debt_usd,
        "excess_usd": excess_usd,
        "excess_jpy": excess_jpy,
        "post_unwind_jpy": jpy_balance + excess_jpy,
    }


def _preview_reduces_margin(preview):
    init_change = preview.get("initMarginChange")
    maint_change = preview.get("maintMarginChange")
    return init_change is not None and maint_change is not None and init_change < 0 and maint_change < 0


def _preview_increases_leveraged_fx(preview):
    message = " ".join(
        str(preview.get(field) or "")
        for field in ("errorString", "rejectReason", "warningText")
    )
    return "increases leveraged fx position" in message.lower()


def choose_deleveraging_action(sell_preview, buy_preview):
    """Return the unique direction that IBKR says reduces margin, else None."""
    sell_reduces = _preview_reduces_margin(sell_preview)
    buy_reduces = _preview_reduces_margin(buy_preview)
    if sell_reduces and not buy_reduces:
        return "SELL"
    if buy_reduces and not sell_reduces:
        return "BUY"

    sell_bad = _preview_increases_leveraged_fx(sell_preview)
    buy_bad = _preview_increases_leveraged_fx(buy_preview)
    if buy_bad and not sell_bad:
        return "SELL"
    if sell_bad and not buy_bad:
        return "BUY"
    return None


def create_usdjpy_contract():
    contract = Contract()
    contract.symbol = "USD"
    contract.secType = "CASH"
    contract.currency = "JPY"
    contract.exchange = "IDEALPRO"
    return contract


def create_fx_order(action, qty_usd, *, what_if=False):
    order = Order()
    order.action = action
    order.totalQuantity = int(qty_usd)
    order.tif = "DAY"
    order.transmit = True
    order.orderType = "MKT"
    order.whatIf = bool(what_if)
    return order


def fetch_usdjpy_rate(ib_client, logger, timeout=12.0):
    """Fetch the current USDJPY rate via IBKR historical bar (MIDPOINT)."""
    contract = create_usdjpy_contract()

    bars = ib_client.request_historical_bars(
        contract, duration="2 D", bar_size="1 day", timeout=timeout
    )
    if not bars:
        logger.error("Could not fetch USDJPY historical bar.")
        return None
    rate = bars[-1]["close"]
    if rate <= 0:
        logger.error(f"Invalid USDJPY rate from IBKR: {rate}")
        return None
    return rate


def preview_unwind_order(ib_client, logger, action, qty_usd, timeout=10.0):
    """Submit a non-executing IBKR what-if order and return OrderState fields."""
    contract = create_usdjpy_contract()
    order = create_fx_order(action, qty_usd, what_if=True)
    order_id = ib_client.allocate_order_id()
    event = threading.Event()
    ib_client.whatif_events[order_id] = event
    logger.info(f"What-if preview #{order_id}: {action} {int(qty_usd)} USD.JPY")
    ib_client.placeOrder(order_id, contract, order)
    if not event.wait(timeout=timeout):
        logger.warning(
            f"Timed out waiting for terminal what-if response for order #{order_id}; "
            f"using latest partial state."
        )
    return ib_client.whatif_states.get(order_id)


def print_preview(label, preview):
    if not preview:
        print(f"  {label}: NO RESPONSE")
        return
    print(
        f"  {label}: init {preview.get('initMarginChange')}, "
        f"maint {preview.get('maintMarginChange')}, "
        f"equity {preview.get('equityWithLoanChange')}, "
        f"fees {preview.get('commissionAndFees')}, "
        f"status={preview.get('status')!r}"
    )
    warning = preview.get("warningText")
    reject = preview.get("rejectReason")
    if warning:
        print(f"    warning: {warning}")
    if reject:
        print(f"    reject: {reject}")


def place_unwind_order(ib_client, logger, qty_usd, action):
    """Place the IBKR-validated unwind order on IDEALPRO."""
    contract = create_usdjpy_contract()
    order = create_fx_order(action, qty_usd, what_if=False)
    order_id = ib_client.allocate_order_id()
    logger.info(
        f"Placing unwind order #{order_id}: {action} {int(qty_usd)} USD.JPY (MKT)"
    )
    ib_client.placeOrder(order_id, contract, order)
    time.sleep(3)  # Brief wait for acknowledgment
    logger.info(f"Order {order_id} submitted. Check TWS/Activity for fill status.")
    return order_id


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unwind excess JPY carry debt to ceiling.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and display the unwind sizing without placing any order.",
    )
    parser.add_argument(
        "--whatif",
        action="store_true",
        help="Compute sizing, preview SELL and BUY via IBKR what-if, then exit.",
    )
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("JPY CARRY DEBT UNWIND — one-off bug-fix remediation")
    logger.info(f"GENERAL_LEVERAGE = {GENERAL_LEVERAGE}x")
    logger.info(f"Corrected ceiling = (LEVERAGE - 1) x NAV")
    logger.info("=" * 60)

    # --- Connect to IBKR ---
    pm = StubPortfolioManager(logger)
    om = StubOrderManager()
    ib_client = WhatIfIBClient(pm, om, logger, client_id=IB_CLIENT_ID + 90)  # offset to avoid collision

    if not ib_client.connect_and_run():
        logger.critical("Failed to connect to IB Gateway. Ensure it is running on "
                        f"{IB_HOST}:{IB_PORT}.")
        return 1

    try:
        # --- Fetch live NAV ---
        nav = pm.get_current_net_liquidation()
        if not nav or nav <= 0:
            logger.error("NAV not available. Wait a few seconds for account data and retry.")
            return 1
        logger.info(f"Live NAV: ${nav:,.2f}")

        # --- Fetch live JPY balance ---
        balances = ib_client.request_currency_balances(currencies=["JPY", "USD"])
        jpy_bal = balances.get("JPY", 0.0)
        usd_bal = balances.get("USD", 0.0)
        logger.info(f"JPY balance: {jpy_bal:,.0f} JPY")
        logger.info(f"USD balance: {usd_bal:,.2f} USD")

        # --- Fetch live USDJPY rate ---
        rate = fetch_usdjpy_rate(ib_client, logger)
        if not rate:
            logger.error("Cannot proceed without USDJPY rate.")
            return 1
        logger.info(f"USDJPY rate (IBKR hist): {rate:.2f}")

        # --- Compute ceiling and excess ---
        sizing = compute_unwind_sizing(nav, GENERAL_LEVERAGE, jpy_bal, rate)
        ceiling_usd = sizing["ceiling_usd"]
        jpy_debt_usd = sizing["jpy_debt_usd"]
        excess_usd = sizing["excess_usd"]
        excess_jpy = sizing["excess_jpy"]
        post_unwind_jpy = sizing["post_unwind_jpy"]

        print()
        print("=" * 60)
        print("UNWIND SIZING SUMMARY")
        print("=" * 60)
        print(f"  NAV:                  ${nav:>12,.2f}")
        print(f"  GENERAL_LEVERAGE:     {GENERAL_LEVERAGE}x")
        print(f"  Corrected ceiling:    ${ceiling_usd:>12,.2f}  = (LEVERAGE-1) x NAV")
        print(f"  JPY debt (native):    {jpy_bal:>12,.0f} JPY")
        print(f"  JPY debt (USD-equiv): ${jpy_debt_usd:>12,.2f}")
        print(f"  USDJPY rate:          {rate:>12,.2f}")
        print(f"  Excess above ceiling: ${excess_usd:>12,.2f}")
        print("=" * 60)

        if excess_usd <= 0:
            print("\nJPY debt is already at or below the corrected ceiling.")
            print("No unwind needed. The code ceiling fix will prevent re-growth.")
            return 0

        print(f"\nTarget unwind size: {int(excess_usd)} USD.JPY notional")
        print(f"  -> reduces JPY debt by ~{excess_jpy:,.0f} JPY")
        print(f"  -> post-unwind JPY debt: ~{post_unwind_jpy:,.0f} JPY (~${ceiling_usd:,.2f})")
        print(f"  -> realizes FX P&L on the unwound portion at {rate:.2f}")

        if args.dry_run:
            print("\n[DRY RUN] No order placed.")
            return 0

        # --- IBKR what-if previews for direction validation ---
        print()
        print("Running IBKR what-if previews for BOTH directions...")
        sell_preview = preview_unwind_order(ib_client, logger, "SELL", excess_usd)
        buy_preview = preview_unwind_order(ib_client, logger, "BUY", excess_usd)
        print()
        print("=" * 60)
        print("IBKR WHAT-IF PREVIEW")
        print("Margin changes should be NEGATIVE for the true de-risking direction.")
        print("=" * 60)
        print_preview(f"SELL {int(excess_usd)} USD.JPY", sell_preview)
        print_preview(f"BUY  {int(excess_usd)} USD.JPY", buy_preview)
        action = choose_deleveraging_action(sell_preview or {}, buy_preview or {})
        if action is None:
            print()
            print("ABORT: IBKR what-if did not identify a unique de-risking direction.")
            print("Do NOT place the unwind order until the direction is manually verified.")
            return 1
        print(f"\nValidated de-risking direction from what-if: {action} {int(excess_usd)} USD.JPY")

        if args.whatif:
            print("\n[WHATIF] Preview complete. No order placed.")
            return 0

        # --- Typed confirmation ---
        print()
        print(
            f"This is a REAL-MONEY FX order: {action} {int(excess_usd)} USD.JPY. "
            "Type UNWIND to confirm, anything else to abort."
        )
        try:
            confirmation = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            confirmation = ""

        if confirmation != "UNWIND":
            print("Aborted. No order placed.")
            return 1

        # --- Place the order ---
        order_id = place_unwind_order(ib_client, logger, excess_usd, action)
        print(f"\nUnwind order #{order_id} submitted.")
        print("Monitor TWS > Activity for fill confirmation.")
        print("After fill, the code ceiling fix (cash_portfolio_manager.py) will")
        print("prevent the JPY debt from re-growing past the corrected ceiling.")

        return 0

    finally:
        ib_client.disconnect_ib()


if __name__ == "__main__":
    raise SystemExit(main())
