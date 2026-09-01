# execution_journal.py
"""
Persisted daily execution journal for algo trading.

Records the full lifecycle of every order — submission, fills, rejections,
cancellations — in a single JSON file per trading day.

Survives process restarts: on startup, the journal is loaded from disk
and used for idempotency checks (was this trade already submitted today?).

Usage:
    journal = ExecutionJournal("execution_journals", logger)
    journal.record_submission(275, "BIL", "BUY", 11, "stock")
    journal.record_event(275, "Filled", 11, 0, 91.44)

Storage:
    execution_journals/2026-04-08_journal.json
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Set


class ExecutionJournal:
    """Persisted daily execution log with full order lifecycle tracking."""

    def __init__(
        self,
        journal_dir: str = "execution_journals",
        logger: Optional[logging.Logger] = None,
        retention_days: int = 30,
    ):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.retention_days = retention_days
        self._lock = Lock()

        # Load or create today's journal
        self._today_str = datetime.now().strftime("%Y-%m-%d")
        self._today_file = self.journal_dir / f"{self._today_str}_journal.json"
        self._data = self._load_or_create()

        # Index: order_id -> index in orders list (for fast event appends)
        self._order_index: Dict[int, int] = {}
        for i, order in enumerate(self._data.get("orders", [])):
            self._order_index[order["order_id"]] = i

        # Cleanup old journals
        self._cleanup_old_journals()

    def _load_or_create(self) -> dict:
        """Load existing journal or create a new one for today."""
        if self._today_file.exists():
            try:
                with open(self._today_file) as f:
                    data = json.load(f)
                n_orders = len(data.get("orders", []))
                self.logger.info(
                    f"Loaded execution journal for {self._today_str}: "
                    f"{n_orders} prior order(s)"
                )
                return data
            except (json.JSONDecodeError, IOError) as e:
                self.logger.error(
                    f"Corrupt journal {self._today_file}: {e}. Creating fresh."
                )

        data = {
            "date": self._today_str,
            "started_at": datetime.now().isoformat(),
            "orders": [],
            "balances": {},
            "summary": {
                "stock_orders": 0,
                "forex_phase1_orders": 0,
                "forex_phase2_orders": 0,
                "settlement_iterations": 0,
                "converged": False,
                "total_fills": 0,
                "total_rejections": 0,
                "total_cancellations": 0,
            },
        }
        self._save(data)
        self.logger.info(f"Created new execution journal for {self._today_str}")
        return data

    # ========================================================================
    # Order submission recording
    # ========================================================================

    def record_submission(
        self,
        order_id: int,
        symbol: str,
        action: str,
        quantity: float,
        order_class: str,
        iteration: Optional[int] = None,
    ):
        """Record a new order submission.

        Args:
            order_id: IBKR order ID.
            symbol: Ticker or forex pair (e.g., 'BIL', 'USD.HKD').
            action: 'BUY' or 'SELL'.
            quantity: Order quantity.
            order_class: 'stock', 'forex_phase1', 'forex_phase2', 'orphan_cleanup'.
            iteration: Settlement iteration number (forex only).
        """
        with self._lock:
            entry = {
                "order_id": order_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "order_class": order_class,
                "submitted_at": datetime.now().isoformat(),
                "events": [],
            }
            if iteration is not None:
                entry["iteration"] = iteration

            self._data["orders"].append(entry)
            self._order_index[order_id] = len(self._data["orders"]) - 1

            # Update summary counters
            summary = self._data["summary"]
            if order_class == "stock" or order_class == "orphan_cleanup":
                summary["stock_orders"] += 1
            elif order_class == "forex_phase1":
                summary["forex_phase1_orders"] += 1
            elif order_class == "forex_phase2":
                summary["forex_phase2_orders"] += 1

            self._save(self._data)

    # ========================================================================
    # Order lifecycle events (fills, cancels, rejects)
    # ========================================================================

    def record_event(
        self,
        order_id: int,
        status: str,
        filled: float = 0.0,
        remaining: float = 0.0,
        avg_price: float = 0.0,
    ):
        """Record a status change event for an order.

        Called from order_manager.update_order_status() on every IBKR callback.

        Args:
            order_id: IBKR order ID.
            status: IBKR status string (Submitted, Filled, Cancelled, etc.).
            filled: Cumulative filled quantity.
            remaining: Remaining quantity.
            avg_price: Average fill price.
        """
        with self._lock:
            idx = self._order_index.get(order_id)
            if idx is None:
                # Order not in today's journal — might be from a prior day
                # or a manual TWS order. Don't record to avoid confusion.
                return

            event = {
                "status": status,
                "filled": filled,
                "remaining": remaining,
                "avg_price": avg_price,
                "at": datetime.now().isoformat(),
            }
            self._data["orders"][idx]["events"].append(event)

            # Update summary counters for terminal states
            summary = self._data["summary"]
            if status == "Filled":
                summary["total_fills"] += 1
            elif status in ("Cancelled", "ApiCancelled"):
                summary["total_cancellations"] += 1
            elif status in ("Rejected", "Inactive"):
                summary["total_rejections"] += 1

            self._save(self._data)

    # ========================================================================
    # Balance snapshots
    # ========================================================================

    def record_balances(self, phase_label: str, balances: Dict[str, float]):
        """Record a currency balance snapshot at a given phase.

        Args:
            phase_label: Label like 'pre_phase1_iter1', 'post_phase2_iter2'.
            balances: Dict of {currency_code: balance}.
        """
        with self._lock:
            self._data["balances"][phase_label] = {
                k: round(v, 2) for k, v in balances.items()
            }
            self._save(self._data)

    def update_summary(self, **kwargs):
        """Update summary fields (e.g., settlement_iterations, converged)."""
        with self._lock:
            self._data["summary"].update(kwargs)
            self._save(self._data)

    # ========================================================================
    # Idempotency queries
    # ========================================================================

    def was_executed_today(self, symbol: str, action: str) -> bool:
        """Check if a symbol+action combination was already submitted today.

        Used by OrderGuard for cross-restart idempotency.
        """
        with self._lock:
            for order in self._data.get("orders", []):
                if order["symbol"] == symbol and order["action"] == action:
                    return True
            return False

    def get_order_ids_today(self) -> Set[int]:
        """Return all order IDs submitted today."""
        with self._lock:
            return {o["order_id"] for o in self._data.get("orders", [])}

    def get_submitted_symbols(self) -> Set[str]:
        """Return all symbols that had orders submitted today."""
        with self._lock:
            return {o["symbol"] for o in self._data.get("orders", [])}

    # ========================================================================
    # Persistence
    # ========================================================================

    def _save(self, data: dict):
        """Atomically persist journal to disk."""
        tmp_file = self._today_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            tmp_file.replace(self._today_file)
        except IOError as e:
            self.logger.error(f"Failed to save journal: {e}")

    def _cleanup_old_journals(self):
        """Delete journal files older than retention_days."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        cleaned = 0
        for f in self.journal_dir.glob("*_journal.json"):
            try:
                date_str = f.stem.replace("_journal", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    f.unlink()
                    cleaned += 1
            except (ValueError, OSError):
                pass
        if cleaned:
            self.logger.info(
                f"Cleaned up {cleaned} old journal file(s) (>{self.retention_days} days)"
            )
