"""Persistent in-algo storage for deferred order submission."""

import json
import time
from pathlib import Path


class PendingOrderBook:
    """Tracks and persists pending LIMIT orders before IBKR submission."""

    def __init__(self, logger, file_path: str):
        self.logger = logger
        self.file_path = Path(file_path)
        self.pending_orders = {}

    def load(self):
        """Load pending orders from disk if available."""
        if not self.file_path.exists():
            return

        try:
            self.pending_orders = json.loads(self.file_path.read_text())
            self.logger.info(
                f"Loaded {len(self.pending_orders)} pending orders from {self.file_path}"
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to load pending orders file {self.file_path}: {e}"
            )
            self.pending_orders = {}

    def persist(self):
        """Persist current pending orders to disk."""
        try:
            self.file_path.write_text(
                json.dumps(self.pending_orders, indent=2, sort_keys=True)
            )
        except Exception as e:
            self.logger.error(
                f"Failed to persist pending orders to {self.file_path}: {e}"
            )

    def replace_with_trades(self, trades_to_execute: dict, exchange_manager):
        """Replace book with fresh trade intents for this cycle."""
        now = time.time()
        updated = {}

        for symbol, quantity in trades_to_execute.items():
            updated[symbol] = {
                "symbol": symbol,
                "quantity": float(quantity),
                "action": "BUY" if quantity > 0 else "SELL",
                "exchange": exchange_manager.get_exchange(symbol),
                "created_at": now,
                "status": "PENDING_SUBMISSION",
                "attempts": 0,
                "last_error": None,
            }

        self.pending_orders = updated
        self.persist()

    def list_by_exchange(self):
        """Return pending orders grouped by exchange."""
        grouped = {}
        for symbol, info in self.pending_orders.items():
            if info.get("status") not in {"PENDING_SUBMISSION", "RETRY"}:
                continue
            exchange = info.get("exchange", "UNKNOWN")
            if exchange not in grouped:
                grouped[exchange] = []
            grouped[exchange].append((symbol, info))
        return grouped

    def mark_submitted(self, symbol: str):
        """Mark symbol as submitted successfully."""
        if symbol in self.pending_orders:
            self.pending_orders[symbol]["status"] = "SUBMITTED"
            self.pending_orders[symbol]["last_error"] = None
            self.persist()

    def mark_retry(self, symbol: str, error_message: str):
        """Mark symbol for retry with error context."""
        if symbol in self.pending_orders:
            self.pending_orders[symbol]["status"] = "RETRY"
            self.pending_orders[symbol]["attempts"] = (
                int(self.pending_orders[symbol].get("attempts", 0)) + 1
            )
            self.pending_orders[symbol]["last_error"] = error_message
            self.persist()

    def mark_failed(self, symbol: str, error_message: str):
        """Mark symbol as failed for this cycle."""
        if symbol in self.pending_orders:
            self.pending_orders[symbol]["status"] = "FAILED"
            self.pending_orders[symbol]["attempts"] = (
                int(self.pending_orders[symbol].get("attempts", 0)) + 1
            )
            self.pending_orders[symbol]["last_error"] = error_message
            self.persist()

    def clear(self):
        """Clear all pending orders and remove persisted file."""
        self.pending_orders = {}
        if self.file_path.exists():
            try:
                self.file_path.unlink()
            except Exception as e:
                self.logger.warning(
                    f"Could not remove pending order file {self.file_path}: {e}"
                )
