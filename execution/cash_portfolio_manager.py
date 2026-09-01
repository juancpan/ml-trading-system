# cash_portfolio_manager.py
#
# Credit Carry Trade Engine
# =========================
# Manages currency credit/debt from stock rebalancing via a two-phase process:
#
# Phase 1 (Exotic Cleanup): Convert exotic currencies to configured majors
#   - Runs after EACH regional stock session (immediate, no ML)
#   - Uses configurable routing (e.g., HUF->EUR, CAD->USD)
#   - LMT/MIDPRICE orders for better fills
#
# Phase 2 (ML-Timed Carry Trade): Convert major currencies to JPY
#   - Runs ONCE daily after ALL regional sessions complete
#   - ML model per Y/JPY pair decides: convert (+1), hold (-1), or partial (proba-based)
#   - Max hold duration guardrail (force-convert after N days)
#   - predict_proba() based conversion sizing
#
# This module replaces forex_manager.py's carry trade logic when
# CASH_REBALANCING_MODE is 'phase1' or 'phase2'.

import json
import logging
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ibapi.contract import Contract
from ibapi.order import Order

from config import CASH_PORTFOLIO_CONFIG

try:
    from config import CASH_ALLOCATION_CONFIG
except ImportError:
    CASH_ALLOCATION_CONFIG = {}

# Revision Protocol — Phase 1.1 signal_history logging for carry signals.
try:
    from signal_history import log_signal as _log_signal_history
except ImportError:
    _log_signal_history = None  # type: ignore

# Pending conversions state file (prevents over-conversion across sessions)
_STATE_DIR = Path(__file__).parent
_PENDING_STATE_FILE = _STATE_DIR / "carry_trade_state.json"
_ALLOCATION_STATE_FILE = _STATE_DIR / "cash_allocation_state.json"


class CashPortfolioManager:
    """Manages currency credit/debt rebalancing and ML-timed JPY carry trade.

    Replaces the static forex_manager carry trade logic with a configurable,
    ML-driven two-phase approach.

    Args:
        ib_client: Connected IBKR client (ib_client_final.IBClient).
        logger: Logger instance.
        strategy_executor: StrategyExecutor for ML signal generation (Phase 2 only).
        exchange_manager: ExchangeManager for currency detection.
        dry_run: If True, log actions but do not place orders.
    """

    def __init__(
        self,
        ib_client,
        logger: logging.Logger,
        strategy_executor=None,
        exchange_manager=None,
        currency_converter=None,
        dry_run: bool = False,
    ):
        self.ib = ib_client
        self.logger = logger
        self.strategy_executor = strategy_executor
        self.exchange_manager = exchange_manager
        self.currency_converter = currency_converter
        self.order_guard = None  # Set via main.py after init
        self.ibkr_data_manager = None  # Set via main.py after init
        self.dry_run = dry_run

        # Unpack config
        self.cfg = CASH_PORTFOLIO_CONFIG
        self.funding_ccy = self.cfg.get("funding_currency", "USD")
        self.carry_ccy = self.cfg.get("carry_currency", "JPY")
        self.carry_pair = self.cfg.get("carry_pair", "USDJPY")
        self.carry_model_cfg = self.cfg.get("carry_model", {})
        self.exotic_routing = self.cfg.get("exotic_routing", {})
        self.min_amount = self.cfg.get("min_amount", 100)
        self.max_hold_days = self.cfg.get("max_hold_days", 30)
        self.order_type = self.cfg.get("forex_order_type", "MKT")
        self.inter_phase_delay = self.cfg.get("inter_phase_delay_seconds", 30)

        # Phase 3: Cash currency allocation config
        self.alloc_cfg = CASH_ALLOCATION_CONFIG
        self.alloc_enabled = self.alloc_cfg.get("enabled", False)
        self.alloc_dry_run = self.alloc_cfg.get("dry_run", True)
        self.alloc_target_weights = self.alloc_cfg.get("target_weights", {})
        self.alloc_threshold = self.alloc_cfg.get("rebalance_threshold_pct", 0.05)
        self.alloc_min_trade = self.alloc_cfg.get("min_trade_usd", 500)
        self.alloc_min_pool = self.alloc_cfg.get("min_pool_usd", 1000)
        self.alloc_settlement_delay = self.alloc_cfg.get("settlement_delay_seconds", 30)
        self.alloc_max_tilt = self.alloc_cfg.get("max_tilt_pp", 0.05)
        self.alloc_ml_models = self.alloc_cfg.get("ml_tilt_models", {})

        # Backward compatibility: majors set for Phase 1 classification
        # Phase 1 treats the funding currency as the only "major"
        self.majors = {self.funding_ccy}

        # Load hold state (tracks how many consecutive days ML said "hold")
        self._hold_state = self._load_hold_state()

    # ========================================================================
    # Public API
    # ========================================================================

    def run_phase1_exotic_cleanup(self, account_values: Dict) -> Dict:
        """Phase 1: Convert exotic currency debt to configured major currencies.

        Called after EACH regional stock session. No ML -- purely mechanical.

        Args:
            account_values: Account values dict from IBKR.

        Returns:
            Results dict with 'executed', 'failed', 'skipped' lists.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PHASE 1: EXOTIC CURRENCY CLEANUP")
        self.logger.info("=" * 60)

        results = {"executed": [], "failed": [], "skipped": []}

        balances = self._get_currency_balances()
        if not balances:
            self.logger.warning("No currency balances available. Skipping Phase 1.")
            return results

        self._log_balance_summary(balances, "Phase 1 - Pre-cleanup")

        # Find exotic currencies with non-trivial balances (debt or credit)
        for ccy, balance in balances.items():
            if ccy in self.majors or ccy == self.carry_ccy:
                continue  # Skip majors and carry currency

            abs_balance = abs(balance)
            if abs_balance < self.min_amount:
                results["skipped"].append(
                    {"currency": ccy, "balance": balance, "reason": "below minimum"}
                )
                continue

            # Determine target major for this exotic
            target_major = self.exotic_routing.get(ccy, "USD")
            if target_major not in self.majors:
                target_major = "USD"  # Safety fallback

            # Determine the forex pair and action
            pair, action, quantity = self._resolve_forex_conversion(
                ccy, target_major, balance
            )

            if pair is None:
                self.logger.warning(
                    f"Cannot find IDEALPRO pair for {ccy} -> {target_major}. Skipping."
                )
                results["skipped"].append(
                    {"currency": ccy, "balance": balance, "reason": "no pair"}
                )
                continue

            self.logger.info(
                f"  {ccy} ({balance:,.2f}) -> {target_major} via {pair} "
                f"({action} {quantity:,.0f})"
            )

            if self.dry_run:
                self.logger.info(f"  [DRY RUN] Would place: {action} {quantity} {pair}")
                results["executed"].append(
                    {
                        "pair": pair,
                        "action": action,
                        "quantity": quantity,
                        "dry_run": True,
                    }
                )
                continue

            # Place the order (Phase 1: exotic → major)
            success = self._place_forex_order(
                pair, action, quantity, order_class="forex_phase1"
            )
            if success:
                results["executed"].append(
                    {"pair": pair, "action": action, "quantity": quantity}
                )
            else:
                results["failed"].append(
                    {"pair": pair, "action": action, "quantity": quantity}
                )

        self.logger.info(
            f"\nPhase 1 complete: {len(results['executed'])} executed, "
            f"{len(results['failed'])} failed, {len(results['skipped'])} skipped"
        )
        return results

    def _compute_carry_ceiling(self, account_values: Dict) -> Optional[Tuple[float, float]]:
        """Compute the JPY carry-debt ceiling and current JPY debt in USD.

        The ceiling is the stock book's actual financing need at configured
        leverage, NOT an independent limit on the carry trade's size:

            ceiling_usd = (GENERAL_LEVERAGE - 1) × NAV

        At GENERAL_LEVERAGE=1.3 and NAV=$11k, ceiling ≈ $3.3k. This caps the
        carry trade to *redenominating* real margin debt into JPY (its purpose),
        not funding an unbounded independent short-JPY/long-USD position.

        Returns:
            (ceiling_usd, current_jpy_debt_usd) or None if NAV/rate unavailable
            (caller must fail-closed in that case).
        """
        try:
            from config import GENERAL_LEVERAGE
        except ImportError:
            self.logger.error(
                "Cannot import GENERAL_LEVERAGE; carry ceiling unavailable."
            )
            return None

        nav = None
        if account_values and isinstance(account_values, dict):
            nl = account_values.get("NetLiquidation")
            if isinstance(nl, dict):
                nav = nl.get("value")
            elif isinstance(nl, (int, float)):
                nav = float(nl)
        if not nav or nav <= 0:
            self.logger.warning(
                "Carry ceiling: NAV unavailable from account_values. "
                "Conversion will be BLOCKED (fail-closed)."
            )
            return None
        try:
            nav = float(nav)
        except (TypeError, ValueError):
            self.logger.warning(f"Carry ceiling: NAV not numeric ({nav!r}).")
            return None

        ceiling_usd = max(0.0, (GENERAL_LEVERAGE - 1.0) * nav)

        jpy_bal = self._get_currency_balances().get(self.carry_ccy, 0.0)
        current_jpy_debt_usd = 0.0
        if jpy_bal < -self.min_amount:
            usd_equiv = self._to_usd_equivalent(self.carry_ccy, jpy_bal)
            if usd_equiv is None or usd_equiv == 0.0:
                self.logger.warning(
                    f"Carry ceiling: JPY balance {jpy_bal:,.0f} could not be "
                    f"converted to USD. Conversion will be BLOCKED (fail-closed)."
                )
                return None
            current_jpy_debt_usd = abs(usd_equiv)

        return (ceiling_usd, current_jpy_debt_usd)

    def run_phase2_carry_trade(self, account_values: Dict) -> Dict:
        """Phase 2: Single-pair carry trade (USD ↔ JPY).

        Daily decision: should USD debt become JPY debt, or vice versa?

        ML signal for USDJPY:
          +1 = USDJPY going up (JPY weakening) → good to owe JPY → CONVERT
          -1 = USDJPY going down (JPY strengthening) → bad to owe JPY → HOLD/REVERT

        Decision matrix (all-or-nothing, daily sweep):
          USD debt + signal +1 → SELL USD.JPY (convert USD debt to JPY debt)
          USD debt + signal -1 → DO NOTHING (keep paying higher USD rate)
          JPY debt + signal +1 → DO NOTHING (JPY debt is favorable, hold)
          JPY debt + signal -1 → SELL USD.JPY (revert JPY debt to USD debt)

        Carry-debt ceiling (bug fix 2026-07-06):
          The sweep is capped at (GENERAL_LEVERAGE - 1) × NAV in USD-equivalent,
          so the carry trade can only redenominate the stock book's real margin
          debt into JPY — never fund an independent unbounded FX position.
          See docs/revision_hypotheses.md (2026-07-06 carry ceiling entry) and
          STRATEGY_MODE.md "JPY Carry Trade — size ceiling".

        Args:
            account_values: Account values dict from IBKR.

        Returns:
            Results dict with 'converted', 'reverted', 'held', 'forced', 'failed'.
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info(
            f"PHASE 2: CARRY TRADE ({self.funding_ccy} ↔ {self.carry_ccy})"
        )
        self.logger.info("=" * 60)

        results = {
            "converted": 0,
            "reverted": 0,
            "held": 0,
            "forced": 0,
            "failed": 0,
        }

        # Refresh balances after Phase 1
        balances = self._get_currency_balances()
        if not balances:
            self.logger.warning("No currency balances available. Skipping Phase 2.")
            return results

        self._log_balance_summary(balances, "Phase 2 - Pre-carry")

        usd_bal = balances.get(self.funding_ccy, 0.0)
        jpy_bal = balances.get(self.carry_ccy, 0.0)
        pair_str = f"{self.funding_ccy}.{self.carry_ccy}"  # USD.JPY

        self.logger.info(
            f"  {self.funding_ccy}: {usd_bal:,.2f}  |  {self.carry_ccy}: {jpy_bal:,.2f}"
        )

        # Safety check: JPY surplus (positive) means prior bug or manual unwind
        if jpy_bal > self.min_amount:
            self.logger.warning(
                f"  *** {self.carry_ccy} SURPLUS DETECTED: {jpy_bal:,.0f} ***\n"
                f"  Expected negative (carry debt) or zero.\n"
                f"  Unwind the {self.carry_ccy} surplus manually in TWS.\n"
                f"  Phase 2 will NOT convert into {self.carry_ccy} surplus."
            )

        # ── Get ML signal ────────────────────────────────────────────
        hold_days = self._hold_state.get(self.carry_pair, 0)
        is_forced = False

        if hold_days >= self.max_hold_days:
            # Safety guardrail: held too long without converting
            self.logger.warning(
                f"  Held {self.funding_ccy} debt for {hold_days} days "
                f"(max={self.max_hold_days}). FORCE-CONVERTING to {self.carry_ccy}."
            )
            signal = 1  # Force convert
            is_forced = True
        else:
            signal = self._get_carry_signal(self.carry_pair, self.carry_model_cfg)
            if signal is None:
                self.logger.error(
                    f"  ML signal generation failed for {self.carry_pair}. "
                    f"Skipping Phase 2."
                )
                results["failed"] += 1
                return results

        self.logger.info(
            f"  {self.carry_pair} signal: {signal:+d} "
            f"({'CONVERT (JPY weakening)' if signal >= 1 else 'HOLD/REVERT (JPY strengthening)'})"
            f"{'  [FORCED]' if is_forced else ''}"
        )

        # ── Execute based on signal + current state ──────────────────

        if signal >= 1:
            # +1: Favorable to hold JPY debt.
            # Sweep any USD debt → JPY debt, capped at the carry ceiling.
            if usd_bal < -self.min_amount:
                ceiling = self._compute_carry_ceiling(account_values)
                if ceiling is None:
                    self.logger.warning(
                        f"  BLOCKED: carry ceiling unavailable (fail-closed). "
                        f"No conversion this cycle."
                    )
                    results["failed"] += 1
                else:
                    ceiling_usd, current_jpy_debt_usd = ceiling
                    headroom_usd = ceiling_usd - current_jpy_debt_usd

                    self.logger.info(
                        f"  Carry ceiling: ${ceiling_usd:,.2f} "
                        f"((LEVERAGE-1)×NAV), "
                        f"current JPY debt: ${current_jpy_debt_usd:,.2f}, "
                        f"headroom: ${headroom_usd:,.2f}"
                    )

                    if headroom_usd <= 0:
                        self.logger.warning(
                            f"  AT CEILING: JPY debt ${current_jpy_debt_usd:,.2f} "
                            f">= ceiling ${ceiling_usd:,.2f}. "
                            f"No further conversion."
                        )
                        results["held"] += 1
                    else:
                        requested_usd = abs(usd_bal)
                        capped_usd = min(requested_usd, headroom_usd)
                        qty = int(capped_usd)
                        if capped_usd < requested_usd:
                            self.logger.warning(
                                f"  CAPPED: requested ${requested_usd:,.2f} "
                                f"→ capped to ${capped_usd:,.2f} "
                                f"(ceiling headroom)."
                            )
                        if qty <= 0:
                            self.logger.info(
                                f"  No headroom for integer conversion (qty={qty}). "
                                f"Holding."
                            )
                            results["held"] += 1
                        else:
                            self.logger.info(
                                f"  CONVERT: SELL {qty:,} {pair_str} "
                                f"({self.funding_ccy} debt → {self.carry_ccy} debt)"
                            )
                            if self.dry_run:
                                self.logger.info(
                                    f"  [DRY RUN] Would SELL {qty:,} {pair_str}"
                                )
                                results["converted"] += 1
                            else:
                                success = self._place_forex_order(
                                    pair_str, "SELL", qty,
                                    order_class="forex_phase2",
                                )
                                if success:
                                    results["forced" if is_forced else "converted"] += 1
                                    self._hold_state[self.carry_pair] = 0
                                else:
                                    results["failed"] += 1
            else:
                self.logger.info(
                    f"  No {self.funding_ccy} debt to convert "
                    f"(balance: {usd_bal:,.2f}). Holding {self.carry_ccy} debt."
                )
                results["held"] += 1

        else:
            # -1: Unfavorable to hold JPY debt.
            # Revert any JPY debt → USD debt.
            if jpy_bal < -self.min_amount:
                # JPY debt exists — revert to USD debt using direction-safe FX logic.
                pair, action, qty = self._resolve_forex_conversion(
                    self.carry_ccy, self.funding_ccy, jpy_bal
                )
                if pair is None or action is None or qty is None:
                    self.logger.error(
                        f"  Cannot revert: failed to resolve {self.carry_ccy} "
                        f"debt conversion to {self.funding_ccy}. "
                        f"Skipping reversion."
                    )
                    results["failed"] += 1
                else:
                    self.logger.info(
                        f"  REVERT: {action} {qty:,} {pair} "
                        f"({self.carry_ccy} debt → {self.funding_ccy} debt)"
                    )

                    if self.dry_run:
                        self.logger.info(f"  [DRY RUN] Would {action} {qty:,} {pair}")
                        results["reverted"] += 1
                    else:
                        success = self._place_forex_order(
                            pair, action, qty, order_class="forex_phase2"
                        )
                        if success:
                            results["reverted"] += 1
                            self._hold_state[self.carry_pair] = 0
                        else:
                            results["failed"] += 1
            elif usd_bal < -self.min_amount:
                # No JPY debt to revert, but USD debt exists — hold it
                self._hold_state[self.carry_pair] = hold_days + 1
                self.logger.info(
                    f"  HOLD: {self.funding_ccy} debt ({usd_bal:,.2f}). "
                    f"Signal says don't convert to {self.carry_ccy}. "
                    f"(day {hold_days + 1}/{self.max_hold_days})"
                )
                results["held"] += 1
            else:
                self.logger.info(
                    f"  No significant debt in either currency. Nothing to do."
                )
                results["held"] += 1

        # Save hold state
        self._save_hold_state()

        self.logger.info(
            f"\nPhase 2 complete: {results['converted']} converted, "
            f"{results['reverted']} reverted, {results['held']} held, "
            f"{results['forced']} forced, {results['failed']} failed"
        )
        return results

    def run_phase3_cash_allocation(self, account_values: Dict) -> Dict:
        """Phase 3: Redistribute positive USD cash across CZK/SGD/CHF/USD basket.

        Runs ONCE daily, AFTER Phase 2 has decided its JPY carry sizing.
        Mechanical drift-based rebalancer — no ML timing gate (unlike Phase 2).
        An optional ML tilt layer (Work Stream C) can adjust target weights via
        predict_proba once each currency's [CCY]JPY model clears validation.

        Safety guarantees:
          - NEVER touches JPY (the carry leg).
          - NEVER runs when USD balance is negative (no diversifying debt).
          - NEVER converts USD debt into other currencies to fund the basket.
          - Reuses _resolve_forex_conversion() for every trade (same direction-safe
            logic Phase 1 uses — avoids reinventing BUY/SELL direction).

        Args:
            account_values: Account values dict from IBKR.

        Returns:
            Results dict with 'rebalanced', 'skipped', 'failed' lists + 'dry_run' flag.
        """
        results: Dict = {
            "rebalanced": [],
            "skipped": [],
            "failed": [],
            "dry_run": self.alloc_dry_run,
            "total_pool_usd": 0.0,
            "weights": {},
        }

        if not self.alloc_enabled:
            self.logger.info("Phase 3 disabled (CASH_ALLOCATION_CONFIG.enabled=False). Skipping.")
            results["skipped"].append({"reason": "disabled"})
            return results

        self.logger.info("\n" + "=" * 60)
        phase_label = "PHASE 3: CASH CURRENCY ALLOCATION"
        if self.alloc_dry_run:
            phase_label += " (DRY RUN — no orders)"
        self.logger.info(phase_label)
        self.logger.info("=" * 60)

        # ── Refresh balances after Phase 2 ───────────────────────────
        balances = self._get_currency_balances()
        if not balances:
            self.logger.warning("No currency balances available. Skipping Phase 3.")
            results["skipped"].append({"reason": "no balances"})
            return results

        # ── Collect the 4 basket currencies ──────────────────────────
        basket_ccys = list(self.alloc_target_weights.keys())  # USD, CZK, SGD, CHF
        native_balances = {ccy: balances.get(ccy, 0.0) for ccy in basket_ccys}

        # ── Convert all to USD-equivalent for weight computation ─────
        usd_equivalents: Dict[str, float] = {}
        for ccy in basket_ccys:
            if ccy == "USD":
                usd_equivalents[ccy] = native_balances[ccy]
            else:
                usd_equivalents[ccy] = self._to_usd_equivalent(ccy, native_balances[ccy])
        total_pool = sum(usd_equivalents.values())
        results["total_pool_usd"] = total_pool

        self.logger.info(f"\n  Total pool (USD-equiv): ${total_pool:,.2f}")
        for ccy in basket_ccys:
            wt = usd_equivalents[ccy] / total_pool * 100 if total_pool > 0 else 0
            self.logger.info(
                f"  {ccy}: {native_balances[ccy]:>12,.2f} native  "
                f"= ${usd_equivalents[ccy]:>10,.2f}  ({wt:.1f}%)"
            )

        # ── Hard gate 1: USD must not be in debt ─────────────────────
        if native_balances.get("USD", 0.0) < 0:
            self.logger.info(
                f"\n  SKIP: USD balance is negative (${native_balances['USD']:,.2f}). "
                f"Phase 3 never diversifies debt. JPY carry trade owns the debt leg."
            )
            results["skipped"].append({
                "reason": "usd_debt",
                "usd_balance": native_balances["USD"],
            })
            self._save_allocation_state(results)
            return results

        # ── Hard gate 2: minimum pool size ───────────────────────────
        if total_pool < self.alloc_min_pool:
            self.logger.info(
                f"\n  SKIP: Total pool ${total_pool:,.2f} below minimum "
                f"${self.alloc_min_pool:,.2f}. Not worth the spread cost."
            )
            results["skipped"].append({
                "reason": "pool_too_small",
                "pool": total_pool,
                "min": self.alloc_min_pool,
            })
            self._save_allocation_state(results)
            return results

        # ── Compute target weights (static + optional ML tilt) ───────
        target_weights = self._compute_target_weights()
        results["weights"] = target_weights

        self.logger.info("\n  Target weights (static + ML tilt):")
        for ccy in basket_ccys:
            self.logger.info(f"    {ccy}: {target_weights[ccy]:.1%}")

        # ── Compute drift and trigger rebalance trades ───────────────
        # USD is the reference (funding) currency — it's never traded against
        # itself. Its weight is the residual left over after CZK/SGD/CHF are
        # bought or sold against it. We still log its drift for observability
        # but never generate a "USD → USD" conversion (the bug that caused the
        # "No security definition for USDUSD" errors on the first shadow run).
        rebalance_trades: List[Dict] = []
        for ccy in basket_ccys:
            actual_weight = usd_equivalents[ccy] / total_pool if total_pool > 0 else 0
            drift = actual_weight - target_weights[ccy]
            drift_usd = drift * total_pool

            self.logger.info(
                f"\n  {ccy}: actual {actual_weight:.1%} vs target "
                f"{target_weights[ccy]:.1%} → drift {drift:+.1%} (${drift_usd:+,.2f})"
            )

            # USD is the reference currency — never trade it against itself.
            # Its allocation is the residual after the other 3 are rebalanced.
            if ccy == self.funding_ccy:
                self.logger.info(
                    f"    Reference currency — drift absorbed as residual "
                    f"(no self-conversion)."
                )
                results["skipped"].append({
                    "currency": ccy,
                    "drift": drift,
                    "drift_usd": drift_usd,
                    "reason": "reference_currency",
                })
                continue

            if abs(drift) <= self.alloc_threshold:
                self.logger.info(f"    Within threshold (±{self.alloc_threshold:.0%}). Skip.")
                results["skipped"].append({
                    "currency": ccy,
                    "drift": drift,
                    "drift_usd": drift_usd,
                    "reason": "within_threshold",
                })
                continue

            if abs(drift_usd) < self.alloc_min_trade:
                self.logger.info(
                    f"    Drift ${abs(drift_usd):,.2f} below min trade "
                    f"${self.alloc_min_trade}. Skip."
                )
                results["skipped"].append({
                    "currency": ccy,
                    "drift": drift,
                    "drift_usd": drift_usd,
                    "reason": "below_min_trade",
                })
                continue

            # Determine trade direction:
            # drift > 0 → overweight → reduce ccy → convert ccy → USD
            # drift < 0 → underweight → acquire ccy → convert USD → ccy
            if drift > 0:
                # Overweight: sell ccy back to USD
                # Pass positive balance in ccy native units → _resolve_forex_conversion
                # will compute the right pair/action/qty to eliminate that credit.
                native_drift = self._usd_to_native(ccy, abs(drift_usd))
                pair, action, qty = self._resolve_forex_conversion(
                    ccy, "USD", native_drift
                )
                trade_type = "reduce"
            else:
                # Underweight: buy ccy using USD
                # Pass positive USD balance → function sells USD to acquire ccy.
                pair, action, qty = self._resolve_forex_conversion(
                    "USD", ccy, abs(drift_usd)
                )
                trade_type = "acquire"

            if pair is None:
                self.logger.warning(
                    f"    Cannot resolve forex conversion for {ccy} drift. Skip."
                )
                results["failed"].append({
                    "currency": ccy,
                    "drift": drift,
                    "reason": "no_pair",
                })
                continue

            self.logger.info(
                f"    {trade_type.upper()}: {action} {qty:,} {pair} "
                f"(${abs(drift_usd):,.2f} drift)"
            )

            rebalance_trades.append({
                "currency": ccy,
                "pair": pair,
                "action": action,
                "quantity": qty,
                "drift_usd": drift_usd,
                "trade_type": trade_type,
            })

        # ── Execute rebalance trades ─────────────────────────────────
        for trade in rebalance_trades:
            if self.alloc_dry_run:
                self.logger.info(
                    f"  [DRY RUN] Would place: {trade['action']} "
                    f"{trade['quantity']:,} {trade['pair']}"
                )
                results["rebalanced"].append({**trade, "dry_run": True})
                continue

            success = self._place_forex_order(
                trade["pair"],
                trade["action"],
                trade["quantity"],
                order_class="forex_phase3",
            )
            if success:
                results["rebalanced"].append(trade)
            else:
                results["failed"].append({
                    "currency": trade["currency"],
                    "pair": trade["pair"],
                    "reason": "order_failed",
                })

        # ── Wait for fills to settle (live mode only) ────────────────
        if results["rebalanced"] and not self.alloc_dry_run:
            self.logger.info(
                f"\n  Waiting {self.alloc_settlement_delay}s for Phase 3 fills..."
            )
            time.sleep(self.alloc_settlement_delay)

        # ── Summary ──────────────────────────────────────────────────
        self.logger.info(
            f"\nPhase 3 complete: {len(results['rebalanced'])} rebalanced, "
            f"{len(results['skipped'])} skipped, {len(results['failed'])} failed"
            f"{' [DRY RUN]' if self.alloc_dry_run else ''}"
        )

        self._save_allocation_state(results)
        return results

    def _compute_target_weights(self) -> Dict[str, float]:
        """Compute target weights: static floor + optional ML tilt, renormalized.

        For each currency with an enabled ML model, the tilt is:
            tilt = (P(ccy/JPY up) - 0.5) * 2 * max_tilt_pp
            adjusted = static_weight + tilt

        All weights are then renormalized to sum to 1.0. Currencies without
        an enabled model use 100% static weight (0% tilt).

        Returns:
            Dict of {currency: weight} summing to 1.0.
        """
        weights = dict(self.alloc_target_weights)

        for ccy, model_cfg in self.alloc_ml_models.items():
            if not model_cfg.get("enabled", False):
                continue
            if ccy not in weights:
                continue

            prob_up = self._get_tilt_probability(ccy, model_cfg)
            if prob_up is None:
                self.logger.info(
                    f"  ML tilt: {ccy} model enabled but no prediction. "
                    f"Using static weight."
                )
                continue

            tilt = (prob_up - 0.5) * 2 * self.alloc_max_tilt
            adjusted = weights[ccy] + tilt
            # Clamp to [0, 1] before renormalization
            weights[ccy] = max(0.0, min(1.0, adjusted))

            self.logger.info(
                f"  ML tilt: {ccy} P(up)={prob_up:.3f} → tilt {tilt:+.1%} "
                f"→ adjusted weight {weights[ccy]:.1%}"
            )

        # Renormalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _get_tilt_probability(
        self, ccy: str, model_cfg: Dict
    ) -> Optional[float]:
        """Get P(ccy/JPY up) from the tilt model for a currency.

        Uses predict_proba() on the GNB model, mirroring the existing
        _get_carry_signal() pattern but returning the probability rather
        than a binary signal.

        Args:
            ccy: Currency code (e.g., 'CZK').
            model_cfg: Model config dict with model_type, strategy_model_path, etc.

        Returns:
            Probability (0.0–1.0) of ccy/JPY going up, or None on failure.
        """
        model_path_str = model_cfg.get("strategy_model_path")
        if not model_path_str:
            return None

        model_path = Path(__file__).parent / model_path_str
        if not model_path.exists():
            self.logger.info(
                f"  ML tilt: {ccy} model not found at {model_path}. Using static."
            )
            return None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Generate features (lagged returns of [CCY]JPY)
            pair_ticker = f"{ccy}JPY"
            features = self._get_live_features(pair_ticker, model_cfg)
            if features is None:
                return None

            # predict_proba returns [[P(class_0), P(class_1)]]
            proba = model.predict_proba(features)
            # class_1 = "up" (signal +1 in training), per the gnb_model convention
            return float(proba[0][1])

        except Exception as e:
            self.logger.warning(f"  ML tilt: {ccy} model prediction failed: {e}")
            return None

    def _to_usd_equivalent(self, ccy: str, native_amount: float) -> float:
        """Convert a native-currency amount to USD-equivalent.

        Uses the same rate-fetching chain as _get_pair_rate().

        Args:
            ccy: Currency code (e.g., 'CZK').
            native_amount: Amount in native currency units.

        Returns:
            USD-equivalent value (float). Returns 0.0 if rate unavailable.
        """
        if abs(native_amount) < 0.01:
            return 0.0

        # For USD-quoted pairs (USD.CZK, USD.SGD, USD.CHF), the rate is
        # "units of ccy per 1 USD". So USD-equiv = native_amount / rate.
        pair_str = f"USD.{ccy}"
        rate = self._get_pair_rate(pair_str)
        if rate and rate > 0:
            return native_amount / rate

        # Fallback: try currency_converter
        if self.currency_converter is not None:
            usd_val = self.currency_converter.convert_to_usd(native_amount, ccy)
            if usd_val is not None:
                return usd_val

        self.logger.warning(
            f"  Cannot convert {native_amount:,.2f} {ccy} to USD — rate unavailable."
        )
        return 0.0

    def _usd_to_native(self, ccy: str, usd_amount: float) -> float:
        """Convert a USD amount to native currency units.

        Args:
            ccy: Currency code (e.g., 'CZK').
            usd_amount: Amount in USD.

        Returns:
            Native currency amount (float). Returns 0.0 if rate unavailable.
        """
        pair_str = f"USD.{ccy}"
        rate = self._get_pair_rate(pair_str)
        if rate and rate > 0:
            return usd_amount * rate

        self.logger.warning(
            f"  Cannot convert ${usd_amount:,.2f} to {ccy} — rate unavailable."
        )
        return 0.0

    def _load_allocation_state(self) -> Dict:
        """Load the Phase 3 allocation state from disk."""
        try:
            if _ALLOCATION_STATE_FILE.exists():
                with open(_ALLOCATION_STATE_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load allocation state: {e}")
        return {}

    def _save_allocation_state(self, results: Dict):
        """Save the Phase 3 allocation state to disk."""
        try:
            state = {
                "last_updated": datetime.now().isoformat(),
                "total_pool_usd": results.get("total_pool_usd", 0.0),
                "weights": results.get("weights", {}),
                "rebalanced_count": len(results.get("rebalanced", [])),
                "skipped_count": len(results.get("skipped", [])),
                "failed_count": len(results.get("failed", [])),
                "dry_run": results.get("dry_run", True),
            }
            with open(_ALLOCATION_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save allocation state: {e}")

    def run_full_rebalancing(self, account_values: Dict) -> Dict:
        """Run the complete cash rebalancing pipeline with iterative convergence.

        Forex conversions are a connected graph: exotic→major changes major
        balances, major→JPY changes JPY debt. A single pass cannot converge
        because Phase 1 fills create new major residuals, and Phase 2 uses
        balances that may not reflect all Phase 1 settlements.

        Solution: iterate Phase 1 + Phase 2 until all non-JPY balances are
        below min_amount, or max iterations reached (typically 2-3).

        Args:
            account_values: Account values dict from IBKR.

        Returns:
            Combined results dict with per-iteration 'phase1'/'phase2' sub-dicts.
        """
        max_iterations = self.cfg.get("max_settlement_iterations", 3)
        results = {"iterations": []}

        for iteration in range(1, max_iterations + 1):
            iter_result = {}
            self.logger.info(
                f"\n{'=' * 60}\n"
                f"CASH SETTLEMENT ITERATION {iteration}/{max_iterations}\n"
                f"{'=' * 60}"
            )

            # Reset OrderGuard forex tracking for this iteration
            if self.order_guard:
                self.order_guard.reset_settlement_cycle(iteration=iteration)

            # Record pre-Phase-1 balances in journal
            if self.order_guard and self.order_guard.journal:
                pre_balances = self._get_currency_balances()
                if pre_balances:
                    self.order_guard.journal.record_balances(
                        f"pre_phase1_iter{iteration}", pre_balances
                    )

            # Phase 1: Exotic cleanup
            iter_result["phase1"] = self.run_phase1_exotic_cleanup(account_values)

            # Wait for Phase 1 fills to settle and refresh balances
            if iter_result["phase1"]["executed"] and not self.dry_run:
                self.logger.info(
                    f"\nWaiting {self.inter_phase_delay}s for Phase 1 fills to settle..."
                )
                time.sleep(self.inter_phase_delay)
                self.logger.info("Refreshing currency balances after Phase 1...")
                self._refresh_balances()

            # Record post-Phase-1 balances
            if self.order_guard and self.order_guard.journal:
                post_p1 = self._get_currency_balances()
                if post_p1:
                    self.order_guard.journal.record_balances(
                        f"post_phase1_iter{iteration}", post_p1
                    )

            # Phase 2: ML-timed carry trade
            iter_result["phase2"] = self.run_phase2_carry_trade(account_values)

            # Record post-Phase-2 balances
            if self.order_guard and self.order_guard.journal:
                post_p2 = self._get_currency_balances()
                if post_p2:
                    self.order_guard.journal.record_balances(
                        f"post_phase2_iter{iteration}", post_p2
                    )

            results["iterations"].append(iter_result)

            # Check convergence: are all non-JPY, non-major exotic balances settled?
            if not self.dry_run and iteration < max_iterations:
                # Wait for Phase 2 fills to settle
                if iter_result["phase2"].get("converted", 0) > 0:
                    self.logger.info(
                        f"\nWaiting {self.inter_phase_delay}s for Phase 2 fills to settle..."
                    )
                    time.sleep(self.inter_phase_delay)

                self._refresh_balances()
                balances = self._get_currency_balances()

                # Check if any exotic currency still has a non-trivial balance
                has_residuals = False
                for ccy, bal in balances.items():
                    if ccy == self.carry_ccy:
                        continue  # JPY is the target, ignore
                    if ccy in self.majors:
                        continue  # Majors handled by Phase 2
                    if abs(bal) >= self.min_amount:
                        has_residuals = True
                        self.logger.info(
                            f"  Residual: {ccy} = {bal:,.2f} "
                            f"(above min_amount={self.min_amount})"
                        )

                if not has_residuals:
                    self.logger.info(
                        f"\nAll exotic balances converged after {iteration} iteration(s)."
                    )
                    break
                else:
                    self.logger.info(
                        f"\nExotic residuals remain. Running iteration {iteration + 1}..."
                    )

        # Summary
        total_p1 = sum(len(it["phase1"]["executed"]) for it in results["iterations"])
        total_p2 = sum(it["phase2"].get("converted", 0) for it in results["iterations"])
        self.logger.info(
            f"Phase 1: {total_p1} orders | "
            f"Phase 2: {total_p2} converted, "
            f"{len(results['iterations'])} iteration(s)"
        )

        # Backward-compatible keys for callers that expect flat structure
        results["phase1"] = results["iterations"][-1]["phase1"]
        results["phase2"] = results["iterations"][-1]["phase2"]

        # ── Phase 3: Cash currency allocation (runs once, after Phase 1+2) ──
        # Layered on top of the carry-trade engine. Redistributes positive USD
        # cash across CZK/SGD/CHF/USD per target weights. Never touches JPY,
        # never runs when USD is in debt. See CASH_ALLOCATION_CONFIG in config.py.
        results["phase3"] = self.run_phase3_cash_allocation(account_values)

        return results

    # ========================================================================
    # Private: Forex conversion resolution
    # ========================================================================

    # IDEALPRO pair conventions — reuses ForexManager's validated mappings.
    # Keys are currencies; values are the correct IDEALPRO pair string and
    # whether the currency is the quote currency in that pair.
    #
    # When routing FROM an exotic TO a major (e.g., HUF debt -> EUR):
    #   - We need to BUY the exotic (cover debt) by SELLing the major pair.
    #   - The pair must be the one IDEALPRO recognizes (e.g., EUR.HUF, not HUF.EUR).
    #
    # These are imported from ForexManager at class level to avoid duplication.
    # Format: {exotic_ccy: {target_major: (idealpro_pair, is_exotic_quote)}}
    # is_exotic_quote=True means the exotic is the QUOTE currency (e.g., USD.HUF -> HUF is quote)

    # Lazy-loaded at first use to avoid circular import issues
    _IDEALPRO_CONVENTIONS = None

    @classmethod
    def _get_idealpro_conventions(cls):
        """Build IDEALPRO pair convention table from ForexManager's mappings."""
        if cls._IDEALPRO_CONVENTIONS is not None:
            return cls._IDEALPRO_CONVENTIONS

        from forex_manager import ForexManager

        conventions = {}

        # From ForexManager.TWO_LEG_PAIRS: exotic currencies quoted vs USD
        # e.g., HUF -> ('USD.HUF', 'SELL', True) means USD is base, HUF is quote
        for ccy, (pair, _action, is_quote) in ForexManager.TWO_LEG_PAIRS.items():
            base, quote = pair.split(".")
            conventions.setdefault(ccy, {})[base] = (pair, is_quote)

        # From ForexManager.JPY_PAIRS: currencies that have direct JPY pairs
        # e.g., USD -> 'USD.JPY', meaning USD is base, JPY is quote
        for ccy, pair in ForexManager.JPY_PAIRS.items():
            base, quote = pair.split(".")
            conventions.setdefault(ccy, {})[quote] = (pair, False)  # ccy is base

        # Add EUR-based exotic pairs (not in ForexManager but valid on IDEALPRO)
        # EUR is base for EU-adjacent: EUR.HUF, EUR.SEK, EUR.CZK, EUR.DKK,
        # EUR.PLN, EUR.NOK, EUR.RON, EUR.CHF
        eur_quoted_exotics = ["HUF", "SEK", "CZK", "DKK", "PLN", "NOK", "RON", "CHF"]
        for ccy in eur_quoted_exotics:
            conventions.setdefault(ccy, {})["EUR"] = (f"EUR.{ccy}", True)

        # EUR.USD: EUR is base, USD is quote
        conventions.setdefault("EUR", {})["USD"] = ("EUR.USD", False)
        # GBP.USD: GBP is base, USD is quote
        conventions.setdefault("GBP", {})["USD"] = ("GBP.USD", False)
        # AUD.USD: AUD is base, USD is quote
        conventions.setdefault("AUD", {})["USD"] = ("AUD.USD", False)
        # NZD.USD: NZD is base, USD is quote
        conventions.setdefault("NZD", {})["USD"] = ("NZD.USD", False)
        # USD.CAD: USD is base, CAD is quote
        conventions.setdefault("CAD", {})["USD"] = ("USD.CAD", True)
        # USD.SGD: USD is base, SGD is quote
        conventions.setdefault("SGD", {})["USD"] = ("USD.SGD", True)
        # USD.HKD: USD is base, HKD is quote
        conventions.setdefault("HKD", {})["USD"] = ("USD.HKD", True)
        # USD.ILS: USD is base, ILS is quote
        conventions.setdefault("ILS", {})["USD"] = ("USD.ILS", True)
        # USD.CHF: USD is base, CHF is quote (vs USD route, not EUR)
        conventions.setdefault("CHF", {}).setdefault("USD", ("USD.CHF", True))

        cls._IDEALPRO_CONVENTIONS = conventions
        return conventions

    def _resolve_forex_conversion(
        self, from_ccy: str, to_ccy: str, balance: float
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Determine the IDEALPRO pair, action, and quantity to convert between two currencies.

        Uses validated IDEALPRO pair conventions (derived from ForexManager's mappings)
        to ensure the correct pair direction is used. IDEALPRO pairs have a fixed
        base/quote convention that cannot be reversed.

        Args:
            from_ccy: Source currency (e.g., 'HUF').
            to_ccy: Target currency (e.g., 'EUR').
            balance: Current balance in from_ccy (negative = debt, positive = credit).

        Returns:
            Tuple of (pair_string, action, quantity) or (None, None, None) if no pair exists.
        """
        amount = abs(balance)
        conventions = self._get_idealpro_conventions()

        # Look up the canonical IDEALPRO pair for from_ccy -> to_ccy
        ccy_conventions = conventions.get(from_ccy, {})
        pair_info = ccy_conventions.get(to_ccy)

        if pair_info is None:
            # Try the reverse: to_ccy -> from_ccy
            reverse_conventions = conventions.get(to_ccy, {})
            reverse_info = reverse_conventions.get(from_ccy)
            if reverse_info is not None:
                # Swap the perspective
                pair_str, is_from_ccy_quote = reverse_info
                is_from_ccy_quote = not is_from_ccy_quote
                pair_info = (pair_str, is_from_ccy_quote)

        if pair_info is None:
            self.logger.warning(
                f"No IDEALPRO pair convention found for {from_ccy} -> {to_ccy}. "
                f"Falling back to {to_ccy}.{from_ccy} (may fail)."
            )
            pair_str = f"{to_ccy}.{from_ccy}"
            # Assume from_ccy is quote (like USD.HUF where HUF is exotic quote)
            is_from_ccy_quote = True
        else:
            pair_str, is_from_ccy_quote = pair_info

        # Determine action based on balance direction and pair convention:
        #
        # Goal: eliminate from_ccy balance (debt or credit).
        #
        # If from_ccy is the QUOTE currency (e.g., HUF in EUR.HUF):
        #   - Debt (balance < 0): We need to receive HUF -> BUY EUR.HUF (buy EUR, receive HUF)
        #     Wait -- BUY EUR.HUF means buy EUR with HUF. That SPENDS HUF (increases debt).
        #     We need SELL EUR.HUF: sell EUR to get HUF (covers HUF debt, creates EUR debt).
        #   - Credit (balance > 0): We want to get rid of HUF -> BUY EUR.HUF (spend HUF to buy EUR)
        #
        # If from_ccy is the BASE currency (e.g., EUR in EUR.USD):
        #   - Debt (balance < 0): We need EUR -> BUY EUR.USD (buy EUR with USD)
        #   - Credit (balance > 0): We have excess EUR -> SELL EUR.USD (sell EUR for USD)

        if is_from_ccy_quote:
            # from_ccy is quote: e.g., HUF in EUR.HUF or CAD in USD.CAD
            if balance < 0:
                # Debt in quote ccy: SELL base to receive quote (covers debt)
                action = "SELL"
            else:
                # Credit in quote ccy: BUY base by spending quote (eliminates credit)
                action = "BUY"
        else:
            # from_ccy is base: e.g., EUR in EUR.USD or GBP in GBP.USD
            if balance < 0:
                # Debt in base ccy: BUY pair to receive base (covers debt)
                action = "BUY"
            else:
                # Credit in base ccy: SELL pair to spend base (eliminates credit)
                action = "SELL"

        # CRITICAL: IBKR IDEALPRO totalQuantity is ALWAYS in BASE currency units.
        # If from_ccy is the quote (e.g., HKD balance -> USD.HKD pair),
        # we must convert the balance from quote units to base units using the
        # exchange rate. Without this, submitting 382,245 HKD as the quantity
        # tells IBKR to buy 382,245 USD (the base), costing ~3M HKD.
        if is_from_ccy_quote:
            rate = self._get_pair_rate(pair_str)
            if rate and rate > 0:
                base_qty = int(amount / rate)
                self.logger.info(
                    f"  Converted quantity: {amount:,.0f} {from_ccy} (quote) "
                    f"/ {rate:.4f} = {base_qty:,} {pair_str.split('.')[0]} (base)"
                )
                return pair_str, action, base_qty
            else:
                self.logger.error(
                    f"Cannot fetch rate for {pair_str} to convert {from_ccy} "
                    f"quantity from quote to base units. SKIPPING to avoid "
                    f"catastrophic quantity mismatch."
                )
                return None, None, None
        else:
            # from_ccy IS the base currency — amount is already in base units
            return pair_str, action, int(amount)

    def _get_pair_rate(self, pair_str: str) -> Optional[float]:
        """Fetch the exchange rate for an IDEALPRO pair (BASE/QUOTE).

        Tries currency_converter (IBKR snapshot + IBKR historical fallback) first,
        then IBKR data manager historical MIDPOINT as last resort.

        Args:
            pair_str: IDEALPRO pair in 'BASE.QUOTE' format (e.g., 'USD.HKD').

        Returns:
            Exchange rate (units of quote per 1 base), or None if unavailable.
        """
        base, quote = pair_str.split(".")

        # Try currency_converter (has IBKR + IBKR historical + Frankfurter fallback chain)
        if self.currency_converter is not None:
            rate = self.currency_converter.fetch_forex_rate(base, quote, timeout=10.0)
            if rate and rate > 0:
                return rate

        # IBKR historical MIDPOINT fallback
        if self.ibkr_data_manager is not None:
            try:
                rate = self.ibkr_data_manager.fetch_forex_rate(base, quote)
                if rate and rate > 0:
                    self.logger.info(
                        f"Fetched {pair_str} rate from IBKR historical: {rate:.4f}"
                    )
                    return rate
            except Exception as e:
                self.logger.warning(
                    f"IBKR historical rate fetch failed for {pair_str}: {e}"
                )

        return None

    # ========================================================================
    # Private: Order execution
    # ========================================================================

    def _create_forex_contract(self, pair: str) -> Contract:
        """Create IBKR Forex contract for IDEALPRO.

        Args:
            pair: Currency pair in 'BASE.QUOTE' format (e.g., 'USD.JPY').

        Returns:
            ibapi Contract configured for forex CASH on IDEALPRO.
        """
        base, quote = pair.split(".")
        contract = Contract()
        contract.symbol = base
        contract.secType = "CASH"
        contract.currency = quote
        contract.exchange = "IDEALPRO"
        return contract

    def _place_forex_order(
        self,
        pair: str,
        action: str,
        quantity: int,
        order_class: str = "forex_phase1",
    ) -> bool:
        """Place a forex conversion order on IDEALPRO.

        Routes through OrderGuard for validation and tracking when available.
        Falls back to direct submission if OrderGuard is not wired.

        Args:
            pair: Currency pair in 'BASE.QUOTE' format.
            action: 'BUY' or 'SELL'.
            quantity: Amount in base currency units.
            order_class: 'forex_phase1' or 'forex_phase2' (for journal tagging).

        Returns:
            True if order was successfully placed, False on error.
        """
        if quantity <= 0:
            self.logger.warning(
                f"Skipping zero/negative quantity order: {action} {quantity} {pair}"
            )
            return False

        try:
            contract = self._create_forex_contract(pair)

            order = Order()
            order.action = action
            order.totalQuantity = quantity
            order.tif = "DAY"
            order.transmit = True

            # IDEALPRO forex CASH contracts support: MKT, LMT, STP, STP LMT.
            # MIDPRICE is NOT supported (error 387: "Unsupported order type").
            order.orderType = "MKT"

            # Route through OrderGuard if available (validates + tracks + journals)
            if self.order_guard:
                return self.order_guard.submit_forex_order(
                    pair,
                    action,
                    quantity,
                    contract,
                    order,
                    order_class=order_class,
                )

            # Fallback: direct submission (backward compatible, no guard)
            order_id = self.ib.nextValidOrderId
            self.ib.nextValidOrderId += 1

            self.logger.info(
                f"  Placing forex order #{order_id}: {action} {quantity} {pair} "
                f"(type={order.orderType})"
            )

            self.ib.placeOrder(order_id, contract, order)

            # Brief delay to avoid pacing violations
            delay = random.uniform(1.5, 2.5)
            time.sleep(delay)

            return True

        except Exception as e:
            self.logger.error(
                f"Error placing forex order {action} {quantity} {pair}: {e}"
            )
            return False

    # ========================================================================
    # Private: ML signal generation
    # ========================================================================

    def _get_carry_signal(self, pair_ticker: str, model_cfg: Dict) -> Optional[int]:
        """Get binary ML carry trade signal for USDJPY.

        Returns +1 (convert USD→JPY) or -1 (hold/revert JPY→USD).

        The model was trained like any other stock signal:
          +1 = USDJPY price going UP (USD strengthens, JPY weakens)
          -1 = USDJPY price going DOWN (JPY strengthens)

        For carry trade interpretation:
          +1 → good to owe JPY (weakening) → CONVERT
          -1 → bad to owe JPY (strengthening) → HOLD/REVERT

        Args:
            pair_ticker: Forex pair ticker (e.g., 'USDJPY').
            model_cfg: Model config dict with model_type, strategy_model_path, etc.

        Returns:
            +1 (convert), -1 (hold/revert), or None on error.
        """
        model_path_str = model_cfg.get("strategy_model_path")
        if not model_path_str:
            self.logger.warning(
                f"  {pair_ticker}: No model path configured. "
                f"Defaulting to signal=+1 (convert)."
            )
            return 1

        model_path = Path(__file__).parent / model_path_str
        if not model_path.exists():
            # Model not yet trained/deployed — default to convert (take the carry).
            # The interest rate differential is a structural advantage; the model
            # only adds timing alpha. When in doubt, take the carry.
            self.logger.warning(
                f"  {pair_ticker}: Carry model NOT FOUND at {model_path}. "
                f"Defaulting to signal=+1 (convert). "
                f"Train and deploy the model for ML-timed carry."
            )
            return 1

        # Try strategy_executor first (handles feature engineering)
        if self.strategy_executor is not None:
            try:
                raw = self.strategy_executor.generate_carry_signal(
                    pair_ticker, model_cfg
                )
                # strategy_executor may return a fraction or a signal
                signal = 1 if raw and raw > 0 else -1
                model_type = model_cfg.get("model_type", "unknown")
                self.logger.info(
                    f"  Carry signal for {pair_ticker}: {signal:+d} "
                    f"({'CONVERT' if signal == 1 else 'HOLD/REVERT'}) "
                    f"(raw: {raw}, model_type: {model_type}, "
                    f"source: strategy_executor)"
                )
                self._log_carry_signal(pair_ticker, model_type, signal, raw)
                return signal
            except Exception as e:
                self.logger.error(
                    f"Strategy executor carry signal failed for {pair_ticker}: {e}"
                )
                # Fall through to direct model loading

        # Direct model loading fallback
        try:
            import warnings

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                for w in caught:
                    self.logger.warning(
                        f"Pickle load warning for {pair_ticker}: {w.message}"
                    )

            # Generate features (lagged returns)
            features = self._get_live_features(pair_ticker, model_cfg)
            if features is None:
                return None

            # Get prediction — binary signal
            prediction = model.predict(features)
            raw_signal = float(prediction[0])
            signal = 1 if raw_signal > 0 else -1

            model_type = model_cfg.get("model_type", "unknown")
            self.logger.info(
                f"  Carry signal for {pair_ticker}: {signal:+d} "
                f"({'CONVERT' if signal == 1 else 'HOLD/REVERT'}) "
                f"(raw: {raw_signal:.6f}, model_type: {model_type}, "
                f"source: direct_load)"
            )
            self._log_carry_signal(pair_ticker, model_type, signal, raw_signal)
            return signal

        except Exception as e:
            self.logger.error(
                f"Error loading/running carry model for {pair_ticker}: {e}"
            )
            import traceback

            self.logger.error(traceback.format_exc())
            return None

    def _log_carry_signal(
        self, pair_ticker: str, model_type: str, signal: int, raw_score: float
    ) -> None:
        """Log carry signal to signal_history.parquet (best-effort, never crashes)."""
        if _log_signal_history is None:
            return
        try:
            _log_signal_history(
                region="CARRY",
                ticker=f"carry:{pair_ticker}",
                model_type=str(model_type),
                strategy_type="carry",
                raw_score=float(raw_score) if raw_score is not None else float("nan"),
                signal=int(signal),
                target_weight=0.0,
                kelly_fraction_used=1.0,
            )
        except Exception as _exc:
            self.logger.debug(
                "signal_history log failed for carry:%s: %s", pair_ticker, _exc
            )

    def _get_live_features(
        self, pair_ticker: str, model_cfg: Dict
    ) -> Optional[np.ndarray]:
        """Generate live features for carry trade model prediction.

        Fetches recent price data from the parquet store and computes lagged returns.

        Args:
            pair_ticker: Forex pair ticker (e.g., 'USDJPY').
            model_cfg: Model config dict.

        Returns:
            Numpy array of shape (1, n_features) or None on error.
        """
        try:
            # Try parquet store first
            from algos.common.market_data_store import MarketDataStore

            store = MarketDataStore()
            yf_ticker = f"{pair_ticker}=X"

            if not store.has_ticker(yf_ticker):
                self.logger.warning(f"No parquet data for {yf_ticker}")
                return None

            # Get recent data (enough for lags + warmup)
            lags = model_cfg.get("lags", 5)
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

            df = store.get_ohlcv(yf_ticker, start_date, end_date)
            if df is None or len(df) < lags + 1:
                self.logger.warning(
                    f"Insufficient data for {yf_ticker}: "
                    f"{len(df) if df is not None else 0} rows (need {lags + 1})"
                )
                return None

            # Compute log returns
            close = df["Close"] if "Close" in df.columns else df["close"]
            returns = np.log(close / close.shift(1)).dropna()

            # Create lagged features (same as gnb_model.py pattern)
            feature_data = {}
            for lag in range(1, lags + 1):
                feature_data[f"lag_{lag}"] = returns.shift(lag)

            features_df = pd.DataFrame(feature_data, index=returns.index).dropna()

            if features_df.empty:
                self.logger.warning(
                    f"No features after lag computation for {yf_ticker}"
                )
                return None

            # Return the most recent row
            return features_df.iloc[[-1]].values

        except Exception as e:
            self.logger.error(f"Error generating live features for {pair_ticker}: {e}")
            return None

    # ========================================================================
    # Private: Currency balance utilities
    # ========================================================================

    def _get_currency_balances(self) -> Dict[str, float]:
        """Get per-currency cash balances from IBKR.

        Returns:
            Dictionary of {currency_code: balance} (negative = debt, positive = credit).
        """
        if hasattr(self.ib, "currency_balances") and self.ib.currency_balances:
            return dict(self.ib.currency_balances)

        self.logger.warning(
            "Currency balances not available from reqAccountSummary. "
            "Requesting with expected currencies..."
        )
        # Try to request them
        try:
            from config import SYMBOLS

            expected = {"USD", self.carry_ccy}
            expected.update(self.majors)
            for ccy in self.exotic_routing.keys():
                expected.add(ccy)

            self.ib.request_currency_balances(currencies=list(expected))
            time.sleep(3)  # Brief wait for response

            if self.ib.currency_balances:
                return dict(self.ib.currency_balances)
        except Exception as e:
            self.logger.error(f"Failed to request currency balances: {e}")

        return {}

    def _refresh_balances(self):
        """Refresh currency balances from IBKR after order execution."""
        try:
            expected = {"USD", self.carry_ccy}
            expected.update(self.majors)
            for ccy in self.exotic_routing.keys():
                expected.add(ccy)

            self.ib.request_currency_balances(currencies=list(expected))
            time.sleep(5)  # Allow time for account update
        except Exception as e:
            self.logger.error(f"Failed to refresh balances: {e}")

    def _log_balance_summary(self, balances: Dict[str, float], phase_label: str):
        """Log a formatted summary table of currency balances.

        Args:
            balances: Currency balance dict.
            phase_label: Label for the log header.
        """
        self.logger.info(f"\n{'-' * 60}")
        self.logger.info(f"CURRENCY BALANCES ({phase_label})")
        self.logger.info(f"{'-' * 60}")
        self.logger.info(
            f"{'Currency':<10} {'Balance':>15} {'Type':<12} {'Routing':<20}"
        )
        self.logger.info(f"{'-' * 60}")

        for ccy in sorted(balances.keys()):
            balance = balances[ccy]
            balance_str = f"{balance:>15,.2f}"

            if ccy == self.carry_ccy:
                ccy_type = "CARRY TGT"
                routing = "—"
            elif ccy in self.majors:
                ccy_type = "MAJOR"
                pair = f"{ccy}.{self.carry_ccy}"
                hold_days = self._hold_state.get(ccy, 0)
                routing = f"-> {pair} (hold:{hold_days}d)"
            elif ccy in self.exotic_routing:
                target = self.exotic_routing[ccy]
                ccy_type = "EXOTIC"
                routing = f"-> {target} (immediate)"
            else:
                ccy_type = "UNKNOWN"
                routing = "-> USD (default)"

            self.logger.info(f"{ccy:<10} {balance_str} {ccy_type:<12} {routing:<20}")

        self.logger.info(f"{'-' * 60}")

    # ========================================================================
    # Private: Hold state persistence
    # ========================================================================

    def _load_hold_state(self) -> Dict[str, int]:
        """Load the hold state from disk (tracks consecutive hold days per currency)."""
        try:
            if _PENDING_STATE_FILE.exists():
                with open(_PENDING_STATE_FILE, "r") as f:
                    state = json.load(f)
                return state.get("hold_days", {})
        except Exception as e:
            self.logger.warning(f"Could not load carry trade state: {e}")
        return {}

    def _save_hold_state(self):
        """Save the hold state to disk."""
        try:
            state = {
                "hold_days": self._hold_state,
                "last_updated": datetime.now().isoformat(),
            }
            with open(_PENDING_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save carry trade state: {e}")


# Add pandas import at module level (used in _get_live_features)
try:
    import pandas as pd
except ImportError:
    pd = None
