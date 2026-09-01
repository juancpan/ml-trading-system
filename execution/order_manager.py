# order_manager.py

import threading
import time
from ibapi.contract import Contract
from ibapi.order import Order
from utils import setup_logger


class OrderManager:
    """Manages the lifecycle and status of all submitted orders."""

    FILLED_STATUSES = {"Filled"}
    CANCELLED_STATUSES = {"ApiCancelled", "Cancelled"}
    REJECTED_STATUSES = {"Rejected", "Inactive"}

    def __init__(self, logger):
        self.logger = logger
        self._lock = threading.Lock()
        self.open_orders = {}
        self.filled_orders = {}
        self.cancelled_orders = {}
        self.rejected_orders = {}

        # Optional: execution journal for persisted lifecycle tracking.
        # Set via order_manager.journal = journal_instance after init.
        self.journal = None

    def track_order(self, order_id: int, contract: Contract, order: Order):
        """Initial tracking of a newly placed order."""
        with self._lock:
            self.open_orders[order_id] = {
                "contract": contract,
                "order": order,
                "status": "PENDING_SUBMIT",
                "action": order.action,
                "totalQuantity": order.totalQuantity,
                "filledQuantity": 0.0,
                "remainingQuantity": order.totalQuantity,
                "avgFillPrice": 0.0,
                "lastFillPrice": 0.0,
                "timestamp": time.time(),
            }
        self.logger.info(
            f"OrderManager tracking new order {order_id} for {contract.symbol}."
        )

    def update_order_status(
        self,
        order_id: int,
        status: str,
        filled: float = None,
        remaining: float = None,
        avg_fill_price: float = None,
        last_fill_price: float = None,
    ):
        """Updates the status and details of an tracked order."""
        with self._lock:
            if order_id not in self.open_orders:
                if (
                    order_id in self.filled_orders
                    or order_id in self.cancelled_orders
                    or order_id in self.rejected_orders
                ):
                    self.logger.debug(
                        f"Duplicate status update for terminal order {order_id}: {status}"
                    )
                    return
                self.logger.warning(
                    f"Received status for untracked order ID: {order_id}. Status: {status}"
                )
                return

            order_info = self.open_orders[order_id]
            order_info["status"] = status
            if filled is not None:
                order_info["filledQuantity"] = filled
            if remaining is not None:
                order_info["remainingQuantity"] = remaining
            if avg_fill_price is not None:
                order_info["avgFillPrice"] = avg_fill_price
            if last_fill_price is not None:
                order_info["lastFillPrice"] = last_fill_price

            self.logger.info(
                f"Order {order_id} status updated to: {status}. "
                f"Filled: {order_info['filledQuantity']}/{order_info['totalQuantity']}"
            )

            if status in self.FILLED_STATUSES:
                self.filled_orders[order_id] = self.open_orders.pop(order_id)
            elif status in self.CANCELLED_STATUSES:
                self.cancelled_orders[order_id] = self.open_orders.pop(order_id)
                self.logger.warning(f"Order {order_id} cancelled by broker/API")
            elif status in self.REJECTED_STATUSES:
                self.rejected_orders[order_id] = self.open_orders.pop(order_id)
                self.logger.warning(
                    f"Order {order_id} reached terminal rejected status: {status}"
                )

            # Record lifecycle event in execution journal (if wired)
            if self.journal:
                try:
                    safe_filled = float(filled) if filled is not None else 0.0
                    safe_remaining = float(remaining) if remaining is not None else 0.0
                    safe_price = (
                        float(avg_fill_price) if avg_fill_price is not None else 0.0
                    )
                    self.journal.record_event(
                        order_id, status, safe_filled, safe_remaining, safe_price
                    )
                except Exception as e:
                    self.logger.debug(
                        f"Journal record_event error for order {order_id}: {e}"
                    )

    def confirm_execution(self, order_id: int, shares: float, price: float):
        """Confirms an execution for a given order."""
        with self._lock:
            if order_id in self.open_orders:
                self.logger.info(
                    f"Execution confirmed for order {order_id}: {shares} shares @ {price}"
                )
            else:
                self.logger.warning(
                    f"Execution received for an order ({order_id}) not currently tracked as open."
                )

    def get_open_orders(self):
        """Returns a copy of the currently open orders dictionary."""
        with self._lock:
            return self.open_orders.copy()

    def get_order_info(self, order_id: int):
        """Returns the latest tracked info for an order from any lifecycle bucket."""
        with self._lock:
            if order_id in self.open_orders:
                return self.open_orders[order_id].copy()
            if order_id in self.filled_orders:
                return self.filled_orders[order_id].copy()
            if order_id in self.cancelled_orders:
                return self.cancelled_orders[order_id].copy()
            if order_id in self.rejected_orders:
                return self.rejected_orders[order_id].copy()
            return None

    def get_order_status(self, order_id: int):
        """Returns the current status of a specific order."""
        info = self.get_order_info(order_id)
        return info.get("status") if info else None

    def is_order_open(self, order_id: int):
        """Checks if an order is still in the open order bucket."""
        with self._lock:
            return order_id in self.open_orders

    def is_order_terminal(self, order_id: int):
        """Checks if an order is in any terminal bucket."""
        with self._lock:
            return (
                order_id in self.filled_orders
                or order_id in self.cancelled_orders
                or order_id in self.rejected_orders
            )

    def is_order_pending(self, order_id: int):
        """Checks if an order is still active/pending."""
        status = self.get_order_status(order_id)
        return status in [
            "PENDING_SUBMIT",
            "PreSubmitted",
            "Submitted",
            "Pending",
            "PendingCancel",
        ]

    def has_pending_order_for_symbol(self, symbol: str) -> bool:
        """Check if there is any open/pending order for the given symbol.

        Searches by contract.symbol across all open orders. Used by OrderGuard
        for pre-flight duplicate detection.

        Args:
            symbol: The IBKR contract symbol (e.g., 'BIL', 'USD', 'EUR').

        Returns:
            True if at least one open order matches the symbol.
        """
        with self._lock:
            for order_info in self.open_orders.values():
                contract = order_info.get("contract")
                if contract and contract.symbol == symbol:
                    return True
            return False

    def get_orders_for_symbol(self, symbol: str) -> list:
        """Return all open order entries matching the given symbol.

        Args:
            symbol: The IBKR contract symbol.

        Returns:
            List of (order_id, order_info) tuples for matching open orders.
        """
        with self._lock:
            results = []
            for oid, info in self.open_orders.items():
                contract = info.get("contract")
                if contract and contract.symbol == symbol:
                    results.append((oid, info.copy()))
            return results
