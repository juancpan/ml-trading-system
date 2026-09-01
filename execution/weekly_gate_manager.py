"""
Weekly Gate Manager for live trading.

Manages the weekly ML gate lifecycle:
  - Monday (or configured rebalance day): generate signals, compute gated
    allocation, persist state to JSON
  - Tue-Fri: load persisted state from JSON
  - Provides get_gated_allocation() for main.py
"""

import json
import logging
import os
import sys
from datetime import date
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Project-level import for gate engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from algos.backtest_code.weekly_gate_engine import compute_gated_allocation


class WeeklyGateManager:
    """Manages weekly gated portfolio allocation for live trading."""

    def __init__(
        self,
        hrp_base_weights: Dict[str, float],
        strategy_executor,
        config: Dict,
    ):
        """
        Parameters
        ----------
        hrp_base_weights : dict
            Static HRP weights {ticker: weight}.
        strategy_executor : StrategyExecutor
            Existing strategy executor instance (for generating ML signals).
        config : dict
            WEEKLY_GATE_CONFIG from config.py.
        """
        self.hrp_weights = hrp_base_weights
        self.executor = strategy_executor
        self.max_weight = config.get("max_weight", 0.25)
        self.min_active = config.get("min_active_tickers", 3)
        self.rebalance_day = config.get("rebalance_day", "Monday")
        self.state_file = config.get(
            "state_file", "execution/weekly_gate_state.json"
        )

        self._current_allocation = None
        self._last_rebalance_date = None

    def get_gated_allocation(
        self, current_date: date, tradeable_symbols: list
    ) -> Dict[str, float]:
        """
        Get the current gated allocation.

        On rebalance day: generate fresh signals and compute new allocation.
        On other days: return the persisted allocation from last rebalance.

        Parameters
        ----------
        current_date : date
            Today's date.
        tradeable_symbols : list
            Symbols that are tradeable today (after holiday filtering).

        Returns
        -------
        dict
            {ticker: weight} for all tickers in HRP (gated-out = 0.0).
        """
        day_name = current_date.strftime("%A")
        is_rebalance_day = day_name == self.rebalance_day

        # Already rebalanced today
        if self._last_rebalance_date == current_date and self._current_allocation:
            return self._current_allocation

        if is_rebalance_day:
            logger.info(
                "[WeeklyGate] Rebalance day (%s). Generating signals...", day_name
            )
            allocation = self._rebalance(tradeable_symbols)
            self._persist_state(current_date, allocation)
            return allocation
        else:
            # Load from persisted state
            state = self._load_state()
            if state:
                logger.info(
                    "[WeeklyGate] Using persisted allocation from %s. "
                    "%s active tickers.",
                    state.get("date", "unknown"),
                    state.get("n_active", "?"),
                )
                return state.get("gated_weights", self.hrp_weights)
            else:
                logger.warning(
                    "[WeeklyGate] No persisted state found. Using base HRP weights."
                )
                return self.hrp_weights

    def _rebalance(self, tradeable_symbols: list) -> Dict[str, float]:
        """Generate signals for all HRP tickers and compute gated allocation."""
        signals = {}
        for ticker in self.hrp_weights:
            if ticker in tradeable_symbols:
                try:
                    sig = self.executor.generate_signal(ticker)
                    signals[ticker] = sig
                except Exception as e:
                    logger.warning(
                        "[WeeklyGate] Signal failed for %s: %s. Gate open.", ticker, e
                    )
                    signals[ticker] = 1
            else:
                # Not tradeable today (holiday). Default to gate open.
                signals[ticker] = 1

        result = compute_gated_allocation(
            self.hrp_weights,
            signals,
            max_weight=self.max_weight,
            min_active_tickers=self.min_active,
        )

        self._current_allocation = result["gated_weights"]
        self._last_rebalance_date = date.today()

        logger.info(
            "[WeeklyGate] Rebalanced: %d active, %d gated out, %.1f%% cash",
            result["n_active"],
            result["n_gated_out"],
            result["cash_weight"] * 100,
        )
        for ticker in sorted(result["gated_out_tickers"]):
            logger.info("  GATED OUT: %s", ticker)

        return result["gated_weights"]

    def _persist_state(self, current_date: date, allocation: Dict[str, float]):
        """Save gated allocation to JSON for use on non-rebalance days."""
        state = {
            "date": current_date.isoformat(),
            "gated_weights": allocation,
            "n_active": sum(1 for w in allocation.values() if w > 0),
        }
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.info("[WeeklyGate] State persisted to %s", self.state_file)
        except Exception as e:
            logger.error("[WeeklyGate] Failed to persist state: %s", e)

    def _load_state(self) -> Optional[Dict]:
        """Load persisted gated allocation from JSON."""
        if not os.path.exists(self.state_file):
            return None
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error("[WeeklyGate] Failed to load state: %s", e)
            return None
