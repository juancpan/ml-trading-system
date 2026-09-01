# order_guard.py
"""
Pre-flight order validation and central gatekeeper.

ALL order submissions (stock + forex) should route through OrderGuard.
It validates orders before submission and tracks them for duplicate prevention.

The guard enforces:
    1. No duplicate orders for the same symbol/pair within a trading cycle
    2. No orders conflicting with pending IBKR orders
    3. No cross-restart duplicate orders (via execution journal)
    4. Forex orders get properly tracked in order_manager (previously bypassed)

Usage:
    guard = OrderGuard(ib_client, order_manager, journal, logger)
    ok = guard.submit_stock_order("BIL", contract, order)
    ok = guard.submit_forex_order("USD.HKD", "BUY", 48755, contract, order)
"""

import logging
import random
import time
from typing import Optional, Tuple

from ibapi.contract import Contract
from ibapi.order import Order


class OrderGuard:
    """Central gatekeeper for all order submissions."""

    def __init__(
        self,
        ib_client,
        order_manager,
        journal,
        logger: Optional[logging.Logger] = None,
    ):
        self.ib = ib_client
        self.order_manager = order_manager
        self.journal = journal
        self.logger = logger or logging.getLogger(__name__)

        # Per-trading-day tracking (reset via reset_trading_day)
        self._submitted_stock_symbols: set = set()

        # Per-settlement-cycle tracking (reset via reset_settlement_cycle)
        self._submitted_forex_pairs: set = set()

        # Current settlement iteration (set by caller)
        self._current_iteration: int = 1

        self.logger.info("OrderGuard initialized")

    # ========================================================================
    # Validation
    # ========================================================================

    def validate_stock_order(
        self, symbol: str, action: str, quantity: float, contract: Contract
    ) -> Tuple[bool, str]:
        """Pre-flight validation for a stock order.

        Returns:
            (True, "OK") if order can proceed,
            (False, reason) if order should be blocked.
        """
        # Check 1: Already submitted this symbol today (in-memory)
        if symbol in self._submitted_stock_symbols:
            return False, f"Already submitted {symbol} this trading day (in-memory)"

        # Check 2: Already submitted in today's journal. The journal persists
        # across both same-session settlement iterations AND process restarts,
        # so a hit here usually just means we already placed this order earlier
        # in the same run (the common case) — not necessarily a restart.
        if self.journal and self.journal.was_executed_today(symbol, action):
            return False, (
                f"Already submitted {action} {symbol} today (journal dedup; "
                f"earlier this session or a prior run)."
            )

        # Check 3: Pending order at IBKR for same symbol
        if self.order_manager and self.order_manager.has_pending_order_for_symbol(
            contract.symbol
        ):
            return False, (
                f"Pending IBKR order already exists for {contract.symbol}. "
                f"Wait for fill/cancel before resubmitting."
            )

        return True, "OK"

    def validate_forex_order(
        self, pair: str, action: str, quantity: int
    ) -> Tuple[bool, str]:
        """Pre-flight validation for a forex order.

        Returns:
            (True, "OK") if order can proceed,
            (False, reason) if order should be blocked.
        """
        pair_key = f"{pair}:{action}"

        # Check 1: Already submitted this pair+action in current settlement cycle
        if pair_key in self._submitted_forex_pairs:
            return False, (
                f"Duplicate forex: {action} {pair} already submitted "
                f"in settlement iteration {self._current_iteration}"
            )

        # Check 2: Pending IBKR order for same pair (base symbol)
        base_symbol = pair.split(".")[0]
        if self.order_manager and self.order_manager.has_pending_order_for_symbol(
            base_symbol
        ):
            # For forex, the contract symbol is the base currency (e.g., "USD" for USD.HKD)
            # This could be a false positive if we have stock + forex for the same symbol,
            # so only warn, don't block
            self.logger.warning(
                f"Note: Pending order exists for base symbol '{base_symbol}'. "
                f"May be a different instrument. Proceeding with forex {pair}."
            )

        # Check 3: Already in today's journal. Persists across same-session
        # settlement iterations AND process restarts; a hit here is normally a
        # repeat attempt within the same run (e.g. clearing a residual balance
        # over multiple iterations), not necessarily a restart.
        if self.journal and self.journal.was_executed_today(pair, action):
            return False, (
                f"Already submitted {action} {pair} today (journal dedup; "
                f"earlier this session or a prior run)."
            )

        return True, "OK"

    # ========================================================================
    # Validated submission
    # ========================================================================

    def submit_stock_order(
        self,
        symbol: str,
        contract: Contract,
        order: Order,
        order_id: int,
        order_class: str = "stock",
    ) -> bool:
        """Validated stock order submission.

        Runs pre-flight checks, then submits via ib_client.place_order()
        (which handles order_manager.track_order internally).

        Args:
            symbol: Ticker symbol (e.g., 'BIL', 'ABCD.TA').
            contract: IBKR Contract object.
            order: IBKR Order object.
            order_id: Pre-allocated order ID.
            order_class: 'stock' or 'orphan_cleanup'.

        Returns:
            True if order was submitted, False if blocked.
        """
        ok, reason = self.validate_stock_order(
            symbol, order.action, order.totalQuantity, contract
        )
        if not ok:
            self.logger.warning(
                f"ORDER BLOCKED: {order.action} {order.totalQuantity} {symbol}: {reason}"
            )
            return False

        # Submit via ib_client.place_order (tracks in order_manager automatically)
        result = self.ib.place_order(contract, order, order_id)

        if result:
            self._submitted_stock_symbols.add(symbol)
            if self.journal:
                self.journal.record_submission(
                    order_id=order_id,
                    symbol=symbol,
                    action=order.action,
                    quantity=float(order.totalQuantity),
                    order_class=order_class,
                )
            self.logger.debug(
                f"OrderGuard: {order.action} {order.totalQuantity} {symbol} submitted (#{order_id})"
            )

        return result

    def submit_forex_order(
        self,
        pair: str,
        action: str,
        quantity: int,
        contract: Contract,
        order: Order,
        order_class: str = "forex_phase1",
    ) -> bool:
        """Validated forex order submission.

        Routes through order_manager.track_order() so forex orders
        are properly tracked (previously bypassed).

        Args:
            pair: Currency pair in 'BASE.QUOTE' format (e.g., 'USD.HKD').
            action: 'BUY' or 'SELL'.
            quantity: Order quantity in base currency units.
            contract: IBKR forex Contract object.
            order: IBKR Order object.
            order_class: 'forex_phase1' or 'forex_phase2'.

        Returns:
            True if order was submitted, False if blocked.
        """
        ok, reason = self.validate_forex_order(pair, action, quantity)
        if not ok:
            self.logger.warning(
                f"FOREX ORDER BLOCKED: {action} {quantity} {pair}: {reason}"
            )
            return False

        # Allocate order ID
        order_id = self.ib.nextValidOrderId
        self.ib.nextValidOrderId += 1

        # Track in order_manager (previously bypassed for forex!)
        if self.order_manager:
            self.order_manager.track_order(order_id, contract, order)

        self.logger.info(
            f"  Placing forex order #{order_id}: {action} {quantity} {pair} "
            f"(type={order.orderType})"
        )

        # Submit to IBKR
        self.ib.placeOrder(order_id, contract, order)

        # Track in guard state
        pair_key = f"{pair}:{action}"
        self._submitted_forex_pairs.add(pair_key)

        # Record in journal
        if self.journal:
            self.journal.record_submission(
                order_id=order_id,
                symbol=pair,
                action=action,
                quantity=quantity,
                order_class=order_class,
                iteration=self._current_iteration,
            )

        # Brief delay to avoid IBKR pacing violations
        time.sleep(random.uniform(1.5, 2.5))

        return True

    # ========================================================================
    # Cycle management
    # ========================================================================

    def reset_settlement_cycle(self, iteration: int = 1):
        """Reset per-settlement-iteration forex tracking.

        Called at the start of each settlement iteration in
        run_full_rebalancing(). Stock tracking is NOT reset here —
        stocks are once-per-day.
        """
        self._submitted_forex_pairs.clear()
        self._current_iteration = iteration
        self.logger.debug(f"OrderGuard: settlement cycle reset (iteration {iteration})")

    def reset_trading_day(self):
        """Reset all per-day tracking.

        Called at the start of each new trading day.
        """
        self._submitted_stock_symbols.clear()
        self._submitted_forex_pairs.clear()
        self._current_iteration = 1
        self.logger.info("OrderGuard: trading day reset")

    # ========================================================================
    # Introspection
    # ========================================================================

    def get_submitted_stock_symbols(self) -> set:
        """Return symbols that have had stock orders submitted today."""
        return self._submitted_stock_symbols.copy()

    def get_submitted_forex_pairs(self) -> set:
        """Return forex pair:action keys submitted in the current settlement cycle."""
        return self._submitted_forex_pairs.copy()
