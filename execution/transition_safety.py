"""
Cross-Market Position Transition Safety Manager

Analyzes and enforces safety constraints when trading across multiple exchanges
with partial market closures. Prevents over-leverage when some positions are
"frozen" (markets closed) while others are tradeable.

Example scenario (MLK Day):
- US closed, EU open
- User holds $100k US positions (frozen - can't sell)
- System wants to BUY $50k EU stocks
- ExcessLiquidity is only $30k
- Without safety check: Would blow up account!
- With safety check: Blocks or scales down the trade

Safety Gates:
1. HARD: Total BUY value <= ExcessLiquidity * threshold
2. HARD: Post-trade leverage <= critical threshold
3. SOFT: Post-trade leverage <= warning threshold (requires confirmation)
4. INFO: Frozen positions > 30% of portfolio (logged)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import date
import logging


class RiskLevel(Enum):
    """Risk assessment levels for cross-market transitions."""
    SAFE = "safe"           # All checks pass, auto-proceed
    INFO = "info"           # Minor concerns, log and proceed
    CAUTION = "caution"     # Moderate risk, log warning
    WARNING = "warning"     # Requires user confirmation
    CRITICAL = "critical"   # Hard block, must scale down or cancel


@dataclass
class TransitionMetrics:
    """
    Comprehensive metrics for cross-market position transition analysis.

    All monetary values are in USD.
    """
    # Date context
    analysis_date: date

    # Market status
    open_exchanges: List[str] = field(default_factory=list)
    closed_exchanges: List[str] = field(default_factory=list)

    # Position breakdown by tradeability
    frozen_positions: Dict[str, float] = field(default_factory=dict)      # symbol -> value (can't trade today)
    tradeable_positions: Dict[str, float] = field(default_factory=dict)   # symbol -> value (can trade today)

    # Trade breakdown
    pending_buys: Dict[str, float] = field(default_factory=dict)          # symbol -> value (BUY orders)
    pending_sells: Dict[str, float] = field(default_factory=dict)         # symbol -> value (SELL orders)
    deferred_trades: Dict[str, float] = field(default_factory=dict)       # symbol -> value (must wait)

    # CRITICAL: Margin metrics from IBKR
    excess_liquidity: float = 0.0           # From IBKR
    available_funds: float = 0.0            # From IBKR
    net_liquidation: float = 0.0            # From IBKR

    # Calculated trade impact
    total_buy_value: float = 0.0            # Sum of all BUY orders
    total_sell_value: float = 0.0           # Sum of all SELL orders
    net_margin_impact: float = 0.0          # buys - sells (margin consumed)

    # Leverage metrics
    current_leverage: float = 0.0
    post_trade_leverage: float = 0.0
    target_leverage: float = 0.0            # From config

    # Portfolio percentages
    frozen_percent: float = 0.0             # Frozen positions as % of portfolio
    trade_percent: float = 0.0              # Trade value as % of portfolio

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.SAFE
    exceeds_excess_liquidity: bool = False  # HARD STOP if True
    exceeds_leverage_threshold: bool = False # WARNING if True
    exceeds_critical_leverage: bool = False  # HARD STOP if True

    # Constraint details
    max_allowed_buy_value: float = 0.0      # Based on excess liquidity
    buy_value_shortfall: float = 0.0        # How much we're over limit


class TransitionSafetyManager:
    """
    Manages cross-market position transition safety analysis.

    Integrates with:
    - PortfolioManager: For positions and account values
    - MarketCalendarManager: For market status
    - ExchangeManager: For symbol->exchange mapping
    """

    def __init__(
        self,
        portfolio_manager,
        market_calendar,
        exchange_manager,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize TransitionSafetyManager.

        Args:
            portfolio_manager: PortfolioManager instance
            market_calendar: MarketCalendarManager instance
            exchange_manager: ExchangeManager instance
            logger: Logger instance
        """
        self.portfolio_manager = portfolio_manager
        self.market_calendar = market_calendar
        self.exchange_manager = exchange_manager
        self.logger = logger or logging.getLogger(__name__)

        # Load config parameters
        self._load_config()

    def _load_config(self):
        """Load safety configuration parameters."""
        from config import (
            TRANSITION_SAFETY_ENABLED,
            TRANSITION_MAX_BUY_PERCENT_OF_EXCESS_LIQ,
            TRANSITION_MIN_EXCESS_LIQ_BUFFER,
            TRANSITION_LEVERAGE_WARNING_THRESHOLD,
            TRANSITION_LEVERAGE_CRITICAL_THRESHOLD,
            TRANSITION_AUTO_PROCEED_IF_ALL_MARKETS_OPEN,
            TRANSITION_AUTO_PROCEED_MAX_TRADE_PERCENT,
            TRANSITION_AUTO_PROCEED_MAX_LEVERAGE,
            GENERAL_LEVERAGE,
        )

        self.enabled = TRANSITION_SAFETY_ENABLED
        self.max_buy_percent = TRANSITION_MAX_BUY_PERCENT_OF_EXCESS_LIQ
        self.min_buffer = TRANSITION_MIN_EXCESS_LIQ_BUFFER
        self.leverage_warning = TRANSITION_LEVERAGE_WARNING_THRESHOLD
        self.leverage_critical = TRANSITION_LEVERAGE_CRITICAL_THRESHOLD
        self.auto_proceed_all_open = TRANSITION_AUTO_PROCEED_IF_ALL_MARKETS_OPEN
        self.auto_proceed_max_trade_pct = TRANSITION_AUTO_PROCEED_MAX_TRADE_PERCENT
        self.auto_proceed_max_leverage = TRANSITION_AUTO_PROCEED_MAX_LEVERAGE
        self.target_leverage = GENERAL_LEVERAGE

    def analyze_transition(
        self,
        analysis_date: date,
        trades: Dict[str, int],
        signals: Dict[str, int]
    ) -> TransitionMetrics:
        """
        Perform comprehensive safety analysis for cross-market transition.

        Args:
            analysis_date: Date of the trading session
            trades: Dict of symbol -> shares_to_trade (positive=BUY, negative=SELL)
            signals: Dict of symbol -> signal value

        Returns:
            TransitionMetrics with complete analysis
        """
        metrics = TransitionMetrics(analysis_date=analysis_date)

        # Get market status
        metrics.open_exchanges = self.market_calendar.get_open_exchanges(analysis_date)
        metrics.closed_exchanges = self.market_calendar.get_closed_exchanges(analysis_date)

        # Get IBKR margin metrics
        metrics.net_liquidation = self.portfolio_manager.get_current_net_liquidation() or 0.0
        metrics.excess_liquidity = float(
            self.portfolio_manager.get_account_value('ExcessLiquidity') or 0.0
        )
        metrics.available_funds = float(
            self.portfolio_manager.get_account_value('AvailableFunds') or 0.0
        )
        metrics.target_leverage = self.target_leverage

        # NOTE: Prices should already be cached from main.py's earlier fetch
        # Do NOT call fetch_latest_prices() here to avoid infinite refetch loops

        # Analyze current positions - classify as frozen or tradeable
        self._analyze_positions(metrics, analysis_date)

        # Analyze proposed trades
        self._analyze_trades(metrics, trades)

        # Calculate leverage impact
        self._calculate_leverage_impact(metrics)

        # Calculate max allowed buy value
        self._calculate_constraints(metrics)

        # Assess overall risk level
        self._assess_risk(metrics)

        return metrics

    def _analyze_positions(self, metrics: TransitionMetrics, analysis_date: date):
        """Classify positions as frozen (closed market) or tradeable (open market)."""
        from config import SYMBOLS

        total_frozen = 0.0
        total_tradeable = 0.0

        for symbol, pos_data in self.portfolio_manager.current_positions.items():
            position = float(pos_data.get('position', 0))
            if position == 0:
                continue

            # CRITICAL: Only analyze positions for symbols in our config
            # Old/orphaned positions (AAPL, TSLA, etc.) would trigger endless price fetches
            if symbol not in SYMBOLS:
                self.logger.debug(f"Skipping orphan position {symbol} (not in SYMBOLS)")
                continue

            # Get USD price for position value (use cached prices only)
            price = self.portfolio_manager.latest_prices_usd.get(symbol, 0.0)
            if price <= 0:
                self.logger.warning(f"No cached price for {symbol}, skipping in analysis")
                continue

            position_value = abs(position * price)

            # Determine if this symbol's exchange is currently open (real-time check)
            _, exchange, _ = self.exchange_manager.parse_symbol(symbol)
            is_tradeable = self.market_calendar.is_open_now(exchange)

            if is_tradeable:
                metrics.tradeable_positions[symbol] = position_value
                total_tradeable += position_value
            else:
                metrics.frozen_positions[symbol] = position_value
                total_frozen += position_value

        # Calculate frozen percentage
        total_portfolio = total_frozen + total_tradeable
        if total_portfolio > 0:
            metrics.frozen_percent = (total_frozen / total_portfolio) * 100

        # Calculate current leverage
        if metrics.net_liquidation > 0:
            metrics.current_leverage = total_portfolio / metrics.net_liquidation

    def _analyze_trades(self, metrics: TransitionMetrics, trades: Dict[str, int]):
        """Analyze proposed trades and classify as buys, sells, or deferred."""
        for symbol, quantity in trades.items():
            if quantity == 0:
                continue

            # Get USD price for trade value (use cached prices only to avoid refetch loop)
            price = self.portfolio_manager.latest_prices_usd.get(symbol, 0.0)
            if price <= 0:
                self.logger.warning(f"No cached price for {symbol}, skipping trade analysis")
                continue

            trade_value = abs(quantity * price)

            # Determine if this symbol's exchange is currently open (real-time check)
            _, exchange, _ = self.exchange_manager.parse_symbol(symbol)
            is_tradeable = self.market_calendar.is_open_now(exchange)

            if not is_tradeable:
                # Market closed - trade must be deferred
                metrics.deferred_trades[symbol] = trade_value * (1 if quantity > 0 else -1)
            elif quantity > 0:
                # BUY order on open market
                metrics.pending_buys[symbol] = trade_value
                metrics.total_buy_value += trade_value
            else:
                # SELL order on open market
                metrics.pending_sells[symbol] = trade_value
                metrics.total_sell_value += trade_value

        # Calculate net margin impact
        # IMPORTANT: Sells don't free margin until they execute
        # On partial market days, we should be conservative
        metrics.net_margin_impact = metrics.total_buy_value - metrics.total_sell_value

        # Calculate trade percentage of portfolio
        if metrics.net_liquidation > 0:
            total_trade_value = metrics.total_buy_value + metrics.total_sell_value
            metrics.trade_percent = (total_trade_value / metrics.net_liquidation) * 100

    def _calculate_leverage_impact(self, metrics: TransitionMetrics):
        """Calculate post-trade leverage."""
        # Current gross exposure
        current_exposure = sum(metrics.frozen_positions.values())
        current_exposure += sum(metrics.tradeable_positions.values())

        # Post-trade exposure
        # Buys add to exposure, sells reduce it (from tradeable positions)
        post_trade_exposure = current_exposure + metrics.net_margin_impact

        if metrics.net_liquidation > 0:
            metrics.post_trade_leverage = post_trade_exposure / metrics.net_liquidation

    def _calculate_constraints(self, metrics: TransitionMetrics):
        """Calculate constraint limits based on excess liquidity."""
        # Maximum allowed buy value
        # Start with percentage of excess liquidity
        max_buy = metrics.excess_liquidity * self.max_buy_percent

        # Subtract minimum buffer
        max_buy = max_buy - self.min_buffer

        # Floor at 0
        metrics.max_allowed_buy_value = max(0.0, max_buy)

        # Calculate shortfall if buys exceed limit
        if metrics.total_buy_value > metrics.max_allowed_buy_value:
            metrics.buy_value_shortfall = metrics.total_buy_value - metrics.max_allowed_buy_value

    def _assess_risk(self, metrics: TransitionMetrics):
        """Assess overall risk level based on all metrics."""
        # Check HARD constraints first

        # CRITICAL: Excess liquidity constraint
        if metrics.excess_liquidity <= 0:
            metrics.risk_level = RiskLevel.CRITICAL
            metrics.exceeds_excess_liquidity = True
            self.logger.critical("ExcessLiquidity <= 0! Cannot execute ANY trades.")
            return

        # Check if all markets are open (no frozen positions risk)
        all_markets_open = len(metrics.closed_exchanges) == 0

        # Apply excess liquidity buffer constraint ONLY when there are frozen positions
        # When all markets are open, let IBKR's own margin system handle the check
        # The conservative buffer was designed to protect against frozen positions
        if metrics.total_buy_value > metrics.max_allowed_buy_value:
            if all_markets_open and self.auto_proceed_all_open:
                # All markets open - skip the conservative buffer check
                # Still log as info but don't block
                self.logger.info(
                    f"All markets open - bypassing conservative buffer check "
                    f"(buy=${metrics.total_buy_value:,.0f} > buffer_limit=${metrics.max_allowed_buy_value:,.0f})"
                )
            else:
                # Some markets closed - apply conservative constraint
                metrics.risk_level = RiskLevel.CRITICAL
                metrics.exceeds_excess_liquidity = True

        # CRITICAL: Leverage constraint (always applies regardless of market status)
        if metrics.post_trade_leverage > self.leverage_critical:
            metrics.risk_level = RiskLevel.CRITICAL
            metrics.exceeds_critical_leverage = True

        # If already critical, don't downgrade
        if metrics.risk_level == RiskLevel.CRITICAL:
            return

        # Check WARNING constraints
        if metrics.post_trade_leverage > self.leverage_warning:
            metrics.risk_level = RiskLevel.WARNING
            metrics.exceeds_leverage_threshold = True
            return

        # Check if frozen positions are significant
        if metrics.frozen_percent > 30:
            if metrics.total_buy_value > 0:
                # Buying while significant positions are frozen
                metrics.risk_level = RiskLevel.WARNING
                return

        # Check CAUTION constraints
        if metrics.frozen_percent > 20:
            metrics.risk_level = RiskLevel.CAUTION
            return

        # Check INFO level
        if metrics.frozen_percent > 0 or len(metrics.closed_exchanges) > 0:
            metrics.risk_level = RiskLevel.INFO
            return

        # All clear
        metrics.risk_level = RiskLevel.SAFE

    def generate_report(self, metrics: TransitionMetrics) -> str:
        """
        Generate a formatted report for logging and user display.

        Args:
            metrics: TransitionMetrics from analyze_transition()

        Returns:
            Formatted multi-line string
        """
        lines = [
            "",
            "=" * 70,
            "CROSS-MARKET POSITION TRANSITION ANALYSIS",
            "=" * 70,
            f"Date: {metrics.analysis_date}",
            f"Risk Level: {self._risk_level_emoji(metrics.risk_level)} {metrics.risk_level.value.upper()}",
            "",
            "MARKET STATUS:",
            f"  Open:   {', '.join(metrics.open_exchanges) if metrics.open_exchanges else 'None'}",
            f"  Closed: {', '.join(metrics.closed_exchanges) if metrics.closed_exchanges else 'None'}",
            "",
            "MARGIN ANALYSIS (CRITICAL):",
            f"  Net Liquidation:     ${metrics.net_liquidation:>12,.2f}",
            f"  Excess Liquidity:    ${metrics.excess_liquidity:>12,.2f}",
            f"  Available Funds:     ${metrics.available_funds:>12,.2f}",
            f"  Total BUY Value:     ${metrics.total_buy_value:>12,.2f}",
            f"  Total SELL Value:    ${metrics.total_sell_value:>12,.2f}",
            f"  Net Margin Impact:   ${metrics.net_margin_impact:>12,.2f}",
            "",
        ]

        # Margin constraint check
        all_markets_open = len(metrics.closed_exchanges) == 0
        buffer_bypassed = (
            metrics.total_buy_value > metrics.max_allowed_buy_value and
            all_markets_open and
            self.auto_proceed_all_open and
            not metrics.exceeds_excess_liquidity
        )

        if metrics.exceeds_excess_liquidity:
            lines.append(
                f"  !!!! EXCEEDS MARGIN LIMIT: "
                f"${metrics.total_buy_value:,.0f} > ${metrics.max_allowed_buy_value:,.0f} max"
            )
            lines.append(
                f"  Shortfall: ${metrics.buy_value_shortfall:,.0f}"
            )
        elif buffer_bypassed:
            lines.append(
                f"  Buffer check bypassed (all markets open) "
                f"(${metrics.total_buy_value:,.0f} > buffer ${metrics.max_allowed_buy_value:,.0f})"
            )
            lines.append(
                f"  Letting IBKR margin system handle constraints"
            )
        else:
            lines.append(
                f"  Within margin limits "
                f"(${metrics.total_buy_value:,.0f} < ${metrics.max_allowed_buy_value:,.0f} max)"
            )

        lines.extend([
            "",
            "LEVERAGE ANALYSIS:",
            f"  Current Leverage:    {metrics.current_leverage:>8.2f}x",
            f"  Post-Trade Leverage: {metrics.post_trade_leverage:>8.2f}x"
            + (f"  <- EXCEEDS WARNING ({self.leverage_warning}x)"
               if metrics.exceeds_leverage_threshold else "")
            + (f"  <- EXCEEDS CRITICAL ({self.leverage_critical}x)"
               if metrics.exceeds_critical_leverage else ""),
            f"  Target Leverage:     {metrics.target_leverage:>8.2f}x",
            "",
        ])

        # Frozen positions
        if metrics.frozen_positions:
            lines.append(f"FROZEN POSITIONS (cannot trade today): {metrics.frozen_percent:.1f}% of portfolio")
            for symbol, value in sorted(metrics.frozen_positions.items(),
                                        key=lambda x: -x[1]):
                lines.append(f"  {symbol:12}: ${value:>12,.2f}")
            lines.append("")

        # Today's executable trades
        if metrics.pending_buys or metrics.pending_sells:
            lines.append("TODAY'S TRADES:")
            for symbol, value in sorted(metrics.pending_buys.items(),
                                        key=lambda x: -x[1]):
                lines.append(f"  BUY  {symbol:12}: +${value:>12,.2f}")
            for symbol, value in sorted(metrics.pending_sells.items(),
                                        key=lambda x: -x[1]):
                lines.append(f"  SELL {symbol:12}: -${value:>12,.2f}")
            lines.append(f"  Net Change:     {'+' if metrics.net_margin_impact >= 0 else ''}${metrics.net_margin_impact:>12,.2f}")
            lines.append("")

        # Deferred trades
        if metrics.deferred_trades:
            lines.append("DEFERRED TRADES (market closed):")
            for symbol, value in metrics.deferred_trades.items():
                action = "BUY" if value > 0 else "SELL"
                lines.append(f"  {action:4} {symbol:12}: ${abs(value):>12,.2f}")
            lines.append("")

        # Risk summary
        lines.append("=" * 70)
        if metrics.risk_level == RiskLevel.CRITICAL:
            lines.append("!!!! CRITICAL: Trade execution BLOCKED")
            if metrics.exceeds_excess_liquidity:
                lines.append(f"     Buy orders exceed margin capacity by ${metrics.buy_value_shortfall:,.0f}")
            if metrics.exceeds_critical_leverage:
                lines.append(f"     Post-trade leverage ({metrics.post_trade_leverage:.2f}x) exceeds critical limit ({self.leverage_critical}x)")
        elif metrics.risk_level == RiskLevel.WARNING:
            lines.append(f"!!!! WARNING: Post-trade leverage ({metrics.post_trade_leverage:.2f}x) exceeds threshold ({self.leverage_warning:.1f}x)")
            lines.append("Manual confirmation required.")
        elif metrics.risk_level == RiskLevel.CAUTION:
            lines.append(f"CAUTION: {metrics.frozen_percent:.1f}% of positions frozen. Proceeding with trades.")
        elif metrics.risk_level == RiskLevel.INFO:
            lines.append("INFO: Some markets closed but trades within normal parameters.")
        else:
            lines.append("SAFE: All checks passed.")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _risk_level_emoji(self, level: RiskLevel) -> str:
        """Get emoji for risk level."""
        return {
            RiskLevel.SAFE: "[OK]",
            RiskLevel.INFO: "[i]",
            RiskLevel.CAUTION: "[!]",
            RiskLevel.WARNING: "[!!]",
            RiskLevel.CRITICAL: "[XXX]"
        }.get(level, "")

    def should_block_execution(self, metrics: TransitionMetrics) -> bool:
        """
        Check if trades should be hard-blocked.

        Args:
            metrics: TransitionMetrics from analyze_transition()

        Returns:
            True if execution should be blocked
        """
        return metrics.risk_level == RiskLevel.CRITICAL

    def should_require_confirmation(self, metrics: TransitionMetrics) -> bool:
        """
        Check if trades require user confirmation.

        Args:
            metrics: TransitionMetrics from analyze_transition()

        Returns:
            True if user confirmation is required
        """
        # Never require confirmation if already blocked
        if self.should_block_execution(metrics):
            return False

        # WARNING level always requires confirmation
        if metrics.risk_level == RiskLevel.WARNING:
            return True

        # Check auto-proceed rules
        if self._can_auto_proceed(metrics):
            return False

        # CAUTION level with significant activity requires confirmation
        if metrics.risk_level == RiskLevel.CAUTION:
            if metrics.total_buy_value > 0 and metrics.frozen_percent > 25:
                return True

        return False

    def _can_auto_proceed(self, metrics: TransitionMetrics) -> bool:
        """Check if we can auto-proceed without confirmation."""
        # All markets open - auto-proceed if configured
        if (self.auto_proceed_all_open and
            len(metrics.closed_exchanges) == 0):
            return True

        # Small trades - auto-proceed
        if metrics.trade_percent < self.auto_proceed_max_trade_pct:
            if metrics.post_trade_leverage < self.auto_proceed_max_leverage:
                return True

        # Only sells - always safe
        if metrics.total_buy_value == 0 and metrics.total_sell_value > 0:
            return True

        return False

    def get_safe_trade_subset(
        self,
        trades: Dict[str, int],
        metrics: TransitionMetrics
    ) -> Dict[str, int]:
        """
        Scale down trades to fit within margin constraints.

        Strategy:
        1. Keep all SELL orders (they free margin)
        2. Scale down BUY orders proportionally to fit

        Args:
            trades: Original trades dict
            metrics: TransitionMetrics from analyze_transition()

        Returns:
            Scaled down trades dict
        """
        if not metrics.exceeds_excess_liquidity:
            return trades  # No scaling needed

        if metrics.max_allowed_buy_value <= 0:
            # Can't do any buys - return only sells
            return {s: q for s, q in trades.items() if q < 0}

        # Calculate scale factor for buys
        if metrics.total_buy_value > 0:
            scale_factor = metrics.max_allowed_buy_value / metrics.total_buy_value
            scale_factor = min(scale_factor, 1.0)  # Never scale up
        else:
            scale_factor = 1.0

        self.logger.info(f"Scaling down BUY orders by factor: {scale_factor:.2%}")

        scaled_trades = {}

        for symbol, quantity in trades.items():
            if quantity < 0:
                # Keep all sells unchanged
                scaled_trades[symbol] = quantity
            elif quantity > 0:
                # Scale down buys
                scaled_qty = int(quantity * scale_factor)
                if scaled_qty > 0:
                    # Apply lot size rounding
                    scaled_qty = self.exchange_manager.round_to_lot_size(symbol, scaled_qty)
                    if scaled_qty > 0:
                        scaled_trades[symbol] = scaled_qty
                        self.logger.info(f"  {symbol}: {quantity} -> {scaled_qty} shares")
                    else:
                        self.logger.info(f"  {symbol}: Scaled to 0, skipping")
                else:
                    self.logger.info(f"  {symbol}: Scaled to 0, skipping")

        return scaled_trades

    def get_sells_only(self, trades: Dict[str, int]) -> Dict[str, int]:
        """
        Extract only SELL orders from trades.

        Args:
            trades: Original trades dict

        Returns:
            Trades dict with only sells
        """
        return {s: q for s, q in trades.items() if q < 0}
