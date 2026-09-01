# config.py

import logging
import pytz  # Will need to install this: pip install pytz

# Interactive Brokers API Connection
IB_HOST = "127.0.0.1"  # Or your remote IP if IB Gateway/TWS is elsewhere
IB_PORT = 4002  # 7497 for paper trading via TWS, 4002 for paper via IB Gateway
IB_CLIENT_ID = 0  # Legacy/default client ID. NOTE: prefer get_client_id(region).

# ----------------------------------------------------------------------------
# Per-region API client IDs (fix for IBKR error 326 "client id already in use")
# ----------------------------------------------------------------------------
# IBKR official docs: a single TWS/Gateway session accepts up to 32 concurrent
# API clients, each of which MUST use a distinct client ID. Regions overlap in
# time (EUROPE runs 17:00 until the gateway drops ~23:45, well past the US
# 23:30 and CANADA 23:35 cron fires), so a single shared client ID (the old
# IB_CLIENT_ID = 0) caused the later regions to be rejected with error 326 and
# silently skip trading (see logs/cron_{US,CANADA}_20260615_233000.log).
#
# Each region therefore gets its own non-zero client ID. We avoid ID 0 for all
# regions: although IBKR recommends ID 0 for manual-order auto-binding, this is
# a headless algo with no manual TWS orders, so that semantic is irrelevant and
# a symmetric non-zero scheme is cleaner. IDs are kept low (1-7) and well clear
# of the preflight probe's ID (see preflight_check.py) and the 32-client limit.
IB_CLIENT_ID_BY_REGION = {
    "US": 1,
    "EUROPE": 2,
    "CANADA": 3,
    "ASIA": 4,
    "MIDDLE_EAST": 5,
    "OCEANIA": 6,
    "INDIA": 7,
    "ALL": 1,
}


def _rotator_allocate(label: str):
    """Import and call the shared client-id rotator, bootstrapping sys.path.

    execution modules typically run with cwd=execution/, so the repo root
    (which holds the `algos` package) is not importable by default. Add it
    before importing. Returns an int client id, or None if the rotator is
    unavailable (caller falls back to the static map).
    """
    import os
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from algos.common.client_id_rotator import allocate_client_id

    return allocate_client_id(host=IB_HOST, port=IB_PORT, label=label)


def get_client_id(region: str) -> int:
    """DEPRECATED static allocator — retained as a fallback only.

    Prefer the shared rotator (algos.common.client_id_rotator.allocate_client_id),
    which assigns a unique, collision-checked id per session and is aware of all
    other live sessions (registry + live 326 probe). main.py now calls the
    rotator directly; this function survives so any remaining caller keeps
    working, returning the legacy per-region id.
    """
    return IB_CLIENT_ID_BY_REGION.get(region, IB_CLIENT_ID)


def allocate_session_client_id(region: str) -> int:
    """Allocate a unique IBKR client id for a trading session via the rotator.

    Falls back to the legacy static per-region id only if the rotator cannot be
    imported (so trading never hard-fails on an import issue). The returned id
    should be released via release_session_client_id() on shutdown.
    """
    try:
        cid = _rotator_allocate(label=f"trading:{region}")
        return cid
    except Exception as e:  # noqa: BLE001 - never block trading on allocation
        logging.getLogger(__name__).warning(
            "Client-id rotator unavailable (%s); falling back to static id for "
            "region %s.", e, region,
        )
        return get_client_id(region)


def release_session_client_id(client_id: int) -> None:
    """Release a rotator-allocated trading client id (best effort)."""
    try:
        import os
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from algos.common.client_id_rotator import release_client_id

        release_client_id(client_id, host=IB_HOST, port=IB_PORT)
    except Exception:
        pass

# Trading Parameters
# Symbols are derived from ASSET_SPECIFIC_CONFIGS
# SYMBOLS will be populated automatically
POSITION_SIZE_UNIT = 100  # For initial simple order, adjust based on strategy output

# ============================================================================
# LEVERAGE CONFIGURATION - Dual Mode System
# ============================================================================
# Options: "portfolio_mode" or "isolated_mode"
# - portfolio_mode: Apply general leverage to entire account, then allocate by weight
#   Formula: target_value = (net_liq * GENERAL_LEVERAGE) * portfolio_weight
# - isolated_mode: Apply per-ticker Kelly fractions (legacy behavior)
#   Formula: target_value = net_liq * portfolio_weight * kelly_fraction
LEVERAGE_MODE = "portfolio_mode"

# Portfolio mode: Single leverage multiplier for entire account
GENERAL_LEVERAGE = 1.3  # 1.3x leverage for entire account

# ============================================================================
# STRATEGY MODE - Dual Strategy System
# ============================================================================
# "daily_binary"  = current system: per-ticker ML signal (+1/-1) daily
# "weekly_gated"  = new system: HRP weights gated by ML signals, weekly rebalance
STRATEGY_MODE = "daily_binary"

# Weekly Gate Configuration (only used when STRATEGY_MODE = "weekly_gated")
WEEKLY_GATE_CONFIG = {
    "max_weight": 0.25,  # Max weight per ticker after redistribution
    "min_active_tickers": 3,  # Floor before going to cash
    "rebalance_day": "Monday",  # Day of week to rebalance
    "state_file": "execution/weekly_gate_state.json",  # Persisted allocation
}

# HRP Base Weights (used when STRATEGY_MODE = "weekly_gated")
# Generated by portimization.py / portfolio_exploration_global.py
HRP_BASE_WEIGHTS = {
    # Populated after running HRP optimization. Example:
    # "BBVA.MC": 0.05, "UCG.MI": 0.04, ...
}

# Portfolio Optimization Weights (relative allocations - should sum to ~1.0)
# These weights are applied to the leveraged capital in portfolio_mode
# or to base capital in isolated_mode

TARGET_ALLOCATION = {
    # EXAMPLE portfolio — placeholder values for the public repo.
    # Replace with your own weights (must sum to ~1.0).
    "SPY": 0.40,
    "TLT": 0.20,
    "GLD": 0.15,
    "BIL": 0.15,
    "QQQ": 0.10,
}

# Asset-Specific Configurations: Strategy Model Path & Kelly Fraction
# Kelly fractions are only used in isolated_mode
# EASY MODEL SWITCHING: Change model_type and strategy_model_path to switch models
#
# STRATEGY TYPES:
# - 'ml_signal': Use ML model predictions (requires model_type and strategy_model_path)
# - 'buy_and_hold': Always hold target position (no ML model needed)
#
# MINIMUM POSITION (Optional):
# - 'min_position_shares': Minimum shares to always hold (even on sell signals)
# - When signal = -1: Sell all except minimum (e.g., 100 shares, min=2 → SELL 98)
# - When signal = +1: Ensure target >= minimum
# - Applies to all strategies (ml_signal and buy_and_hold)
# - Must be compatible with exchange lot sizes (e.g., Tokyo requires multiples of 100)

ASSET_SPECIFIC_CONFIGS = {
    # EXAMPLE configurations — placeholder tickers/models for the public repo.
    "SPY": {
        "kelly_fraction": 2.0,
        "strategy_type": "ml_signal",
        "model_type": "svm_optimized",
        "strategy_model_path": "strategy_models/SPY_trading_model_svm.pkl",
        "sequence_length": 5,
        "lags": 5,
        "min_position_shares": None,
    },
    "TLT": {
        "kelly_fraction": 2.0,
        "strategy_type": "ml_signal",
        "model_type": "lstm_optimized",
        "strategy_model_path": "strategy_models/TLT_trading_model_lstm_optimized.pkl",
        "sequence_length": 5,
        "lags": 5,
        "min_position_shares": None,
    },
    "GLD": {
        "kelly_fraction": 2.0,
        "strategy_type": "buy_and_hold",
        "lags": 5,
    },
    "BIL": {
        "kelly_fraction": 2.0,
        "strategy_type": "buy_and_hold",
        "lags": 5,
    },
    "QQQ": {
        "kelly_fraction": 2.0,
        "strategy_type": "ml_signal",
        "model_type": "xgb_optimized",
        "strategy_model_path": "strategy_models/QQQ_trading_model_xgb.pkl",
        "sequence_length": 5,
        "lags": 5,
        "min_position_shares": None,
    },
}

# Derive SYMBOLS list from ASSET_SPECIFIC_CONFIGS keys
SYMBOLS = list(ASSET_SPECIFIC_CONFIGS.keys())

# Blacklist - Symbols to never trade (e.g., promotional shares, restricted stocks)
# These positions will be ignored in all trading operations
# Note: use min_position_shares to keep a small floor position on any ticker
BLACKLISTED_SYMBOLS = []  # Empty - all symbols in SYMBOLS list are tradable

# ============================================================================
# JPY CARRY TRADE CONFIGURATION (LEGACY)
# ============================================================================
# After stock trades are executed, convert all non-JPY debt to JPY debt
# This creates a JPY carry trade position (short JPY, long other currencies)
# NOTE: When CASH_REBALANCING_MODE is 'phase1' or 'phase2', these legacy
# settings are ignored and CASH_PORTFOLIO_CONFIG takes precedence.
ENABLE_JPY_CARRY_TRADE = True  # Set to True to enable JPY carry trade strategy
JPY_CARRY_TRADE_MIN_DEBT = 100  # Only execute if debt > $100 equivalent

# ============================================================================
# CASH PORTFOLIO CONFIGURATION (CARRY TRADE ENGINE)
# ============================================================================
# The cash portfolio operates independently from the stock/ETF portfolio.
# It manages currency credit/debt from stock rebalancing.
#
# Execution modes (set via CASH_REBALANCING_MODE):
#   'legacy'  - Use old forex_manager.py (convert everything to JPY immediately)
#   'phase1'  - Exotic cleanup only (configurable routing, static JPY conversion)
#   'phase2'  - Exotic cleanup + ML-timed carry trade (Y -> JPY via ML signals)
#
# Phase 1 runs after EACH regional stock session (immediate exotic cleanup).
# Phase 2 runs ONCE daily after ALL regional sessions complete.
# ============================================================================

CASH_REBALANCING_MODE = "phase2"  # Options: 'legacy', 'phase1', 'phase2'

CASH_PORTFOLIO_CONFIG = {
    # Enable/disable the entire cash portfolio engine
    "enabled": True,
    # ========================================================================
    # CARRY TRADE: Single-pair debt optimization (USD ↔ JPY)
    # ========================================================================
    # The carry trade exploits the interest rate differential:
    #   - JPY margin rate: ~1-2% (BOJ)
    #   - USD margin rate: ~5-6% (Fed Funds)
    # By converting USD debt to JPY debt, you pay lower interest.
    #
    # Phase 1: Consolidate ALL exotic currency debts into USD.
    # Phase 2: Decide daily whether USD debt should become JPY debt (or revert).
    #
    # ML signal interpretation for USDJPY:
    #   +1 = USDJPY going up (JPY weakening) → good to owe JPY → CONVERT
    #   -1 = USDJPY going down (JPY strengthening) → bad to owe JPY → HOLD/REVERT
    # ========================================================================
    # The funding currency: all debts consolidate here after Phase 1
    "funding_currency": "USD",
    # The carry currency: borrow this instead of USD for lower interest
    "carry_currency": "JPY",
    # The single carry pair (funding_currency + carry_currency)
    "carry_pair": "USDJPY",
    # ML model for the carry pair signal
    "carry_model": {
        "model_type": "gnb",
        "strategy_model_path": "strategy_models/carry_USDJPY_model_gnb.pkl",
        "scaler_path": "strategy_models/carry_USDJPY_scaler.pkl",
        "lags": 5,
    },
    # ========================================================================
    # PHASE 1: Exotic currency routing (all → USD)
    # ========================================================================
    # Every non-USD, non-JPY currency routes to USD via IDEALPRO.
    # Phase 1 runs after each regional stock session completes.
    "exotic_routing": {
        # European currencies → USD
        "EUR": "USD",  # Euro
        "GBP": "USD",  # British Pound
        "CHF": "USD",  # Swiss Franc
        "SEK": "USD",  # Swedish Krona
        "NOK": "USD",  # Norwegian Krone
        "DKK": "USD",  # Danish Krone
        "CZK": "USD",  # Czech Koruna
        "PLN": "USD",  # Polish Zloty
        "HUF": "USD",  # Hungarian Forint
        "RON": "USD",  # Romanian Leu
        # Americas → USD
        "CAD": "USD",  # Canadian Dollar
        "MXN": "USD",  # Mexican Peso
        # Asia-Pacific → USD
        "AUD": "USD",  # Australian Dollar
        "NZD": "USD",  # New Zealand Dollar
        "SGD": "USD",  # Singapore Dollar
        "HKD": "USD",  # Hong Kong Dollar
        "INR": "USD",  # Indian Rupee
        # Middle East / Africa → USD
        "ILS": "USD",  # Israeli Shekel
        "SAR": "USD",  # Saudi Riyal
        "ZAR": "USD",  # South African Rand
        "TRY": "USD",  # Turkish Lira
    },
    # ========================================================================
    # Operational parameters
    # ========================================================================
    # Minimum balance to trigger a forex conversion (in native currency units)
    "min_amount": 100,
    # Maximum consecutive days to hold USD debt without converting to JPY.
    # Safety guardrail: if ML says 'hold' for too long, force-convert.
    "max_hold_days": 30,
    # Forex order type (MKT recommended — IDEALPRO spreads are tight)
    "forex_order_type": "MKT",
    # Seconds to wait between Phase 1 and Phase 2 for fills to settle
    "inter_phase_delay_seconds": 30,
    # Maximum settlement iterations (Phase 1 + Phase 2 loop)
    "max_settlement_iterations": 3,
}

# ============================================================================
# CASH ALLOCATION CONFIGURATION (Phase 3 — currency diversification)
# ============================================================================
# Layered ON TOP of the carry-trade engine above. Phase 3 runs after each
# regional session's Phase 1+2 (same invocation pattern as Phase 2 — fires on
# each regional cron: EUROPE 17:00, US 23:30, CANADA 23:35 UTC). The 5% drift
# threshold is self-limiting across same-day runs: once CZK/SGD/CHF reach
# target weights, subsequent same-day invocations produce no new trades.
#
# Architectural separation from the carry trade:
#   - Phase 3 NEVER touches JPY (the carry leg) — only USD/CZK/SGD/CHF.
#   - Phase 3 NEVER runs when USD balance is negative (no diversifying debt).
#   - Phase 3 NEVER converts USD debt into other currencies to "fund" the basket.
#
# Static weights below are the floor; an optional ML tilt layer (Work Stream C)
# can adjust each currency's weight by ±max_tilt_pp once that currency's
# [CCY]JPY model clears WFOV + DSR≥0.5 + beats the static-weight benchmark
# out-of-sample. Until a model clears, that currency uses 100% static weight.
#
# Inverse-vol weights derived from full-period annualized vol in the basket-hedge
# research (USD 9.58%, CZK 8.95%, SGD 7.40%, CHF 7.39%):
#   weight[ccy] = (1/vol[ccy]) / sum(1/vol[*])
CASH_ALLOCATION_CONFIG = {
    # Enable/disable the Phase 3 allocator
    "enabled": True,
    # Shadow mode: if True, log decisions but place NO orders. Set to False
    # only after a successful shadow-period review. Consistent with the
    # Research Manual's paper-trade-before-capital principle even though
    # this subsystem sits outside REVISION_POLICY.md's literal scope.
    "dry_run": True,
    # ========================================================================
    # Target weights (static floor)
    # ========================================================================
    "target_weights": {
        "USD": 0.173,  # HRP-derived (validated 2026-07-03)
        "CZK": 0.218,  # HRP-derived
        "SGD": 0.290,  # HRP-derived
        "CHF": 0.320,  # HRP-derived
    },
    # ========================================================================
    # ML tilt layer (Work Stream C — inactive until models clear validation)
    # ========================================================================
    # Per-currency GNB model predicting [CCY]JPY direction. Tilt is sized via
    # predict_proba per Phase 2's documented-but-unimplemented sizing concept:
    #   tilt_pct[ccy] = (P(ccy/JPY up) - 0.5) * 2 * max_tilt_pp
    #   adjusted_weight[ccy] = static_weight[ccy] + tilt_pct[ccy]
    #   → renormalize all 4 weights to sum to 1.0
    # A model entry's "enabled" flag is flipped to True only after that
    # currency's model has cleared WFOV + DSR≥0.5 + beat-static-benchmark.
    "max_tilt_pp": 0.05,  # ±5 percentage points
    "ml_tilt_models": {
        "CZK": {
            "enabled": False,  # flip True only after WS-C validation passes
            "model_type": "gnb",
            "strategy_model_path": "strategy_models/tilt_CZKJPY_model_gnb.pkl",
            "scaler_path": "strategy_models/tilt_CZKJPY_scaler.pkl",
            "lags": 5,
        },
        "SGD": {
            "enabled": False,
            "model_type": "gnb",
            "strategy_model_path": "strategy_models/tilt_SGDJPY_model_gnb.pkl",
            "scaler_path": "strategy_models/tilt_SGDJPY_scaler.pkl",
            "lags": 5,
        },
        "CHF": {
            "enabled": False,
            "model_type": "gnb",
            "strategy_model_path": "strategy_models/tilt_CHFJPY_model_gnb.pkl",
            "scaler_path": "strategy_models/tilt_CHFJPY_scaler.pkl",
            "lags": 5,
        },
    },
    # ========================================================================
    # Operational parameters
    # ========================================================================
    # Drift threshold: |actual_weight - target_weight| > this triggers a
    # rebalance trade for that currency. Wider than the stock book's 1%
    # because FX spread costs matter more per-trade than for equities.
    "rebalance_threshold_pct": 0.05,  # 5%
    # Minimum trade size in USD-equivalent. Below this, skip the trade even
    # if drift is over threshold — not worth the spread cost.
    "min_trade_usd": 500,
    # Minimum total pool size in USD-equivalent. Below this, skip Phase 3
    # entirely — diversification isn't worth the operational overhead.
    "min_pool_usd": 1000,
    # Seconds to wait for Phase 3 fills to settle before final balance snapshot.
    "settlement_delay_seconds": 30,
}

# ============================================================================
# MANUAL MARKET CLOSURE OVERRIDES
# ============================================================================
# Override exchange_calendars for unexpected market closures (emergency closures,
# typhoon days, etc.) that aren't in the standard calendar.
# Format: {exchange_code: ['YYYY-MM-DD', ...]}
MANUAL_MARKET_CLOSURES = {
    # Example: Unexpected Tokyo closure
    # 'TSEJ': ['2026-01-20'],
    # Example: London closure
    # 'LSE': ['2026-04-15'],
}

# ============================================================================
# CROSS-MARKET TRANSITION SAFETY
# ============================================================================
# Safety gates to prevent over-leverage when trading across multiple exchanges
# with partial market closures (e.g., MLK Day: US closed, EU open)
TRANSITION_SAFETY_ENABLED = (
    False  # Temporarily disabled - re-enable after fixing margin check bug
)

# HARD LIMITS (will block execution)
TRANSITION_MAX_BUY_PERCENT_OF_EXCESS_LIQ = 0.80  # Max 80% of ExcessLiquidity
TRANSITION_MIN_EXCESS_LIQ_BUFFER = 10000  # Keep $10k minimum buffer

# SOFT LIMITS (will warn and require confirmation)
TRANSITION_LEVERAGE_WARNING_THRESHOLD = 2.0  # Warn if leverage > 2.0x
TRANSITION_LEVERAGE_CRITICAL_THRESHOLD = 2.5  # Block if leverage > 2.5x

# AUTO-PROCEED RULES (bypass confirmation)
TRANSITION_AUTO_PROCEED_IF_ALL_MARKETS_OPEN = True
TRANSITION_AUTO_PROCEED_MAX_TRADE_PERCENT = (
    5.0  # Auto-proceed if trades < 5% of portfolio
)
TRANSITION_AUTO_PROCEED_MAX_LEVERAGE = 1.9  # Auto-proceed if leverage stays < 1.9x

# CONFIRMATION SETTINGS
TRANSITION_CONFIRMATION_TIMEOUT = 300  # 5 minutes
TRANSITION_DEFAULT_ON_TIMEOUT = (
    "cancel"  # Safe default: "cancel", "proceed", "sells_only"
)

# ============================================================================
# STRATEGY TYPE CONFIGURATION
# ============================================================================
# Valid strategy types for each ticker
VALID_STRATEGY_TYPES = ["ml_signal", "buy_and_hold"]

# Default strategy type if not specified in ASSET_SPECIFIC_CONFIGS
DEFAULT_STRATEGY_TYPE = "ml_signal"

# Rebalancing Thresholds
REBALANCE_THRESHOLD_PERCENT = 0.01  # 1% deviation
MIN_TRADE_SHARES = 1

# ML Signal Override Settings
# When True, ignores ML signal conflicts for entering initial positions (0 -> target)
# This helps when ML models fail to load or give conflicting signals on startup
OVERRIDE_ML_FOR_INITIAL_ENTRY = False  # Changed to False - only follow ML signals

# Daily Trading Schedule (in EST - Eastern Standard Time / New York Time)
# The time at which the daily trading logic (data fetch, strategy, rebalance) will be attempted
# 5:00 PM EST is optimal because:
# - All markets (US, London, Tokyo) have closed and fresh data is available
# - Before Tokyo opens (7 PM EST) so Tokyo orders execute immediately when market opens
# - All orders execute within 24 hours (no IBKR midnight cancellation issues)
# TRADING_HOUR_EST = 17  # 5 PM EST (after US close at 4 PM, before Tokyo open at 7 PM)
TRADING_HOUR_EST = 9
TRADING_MINUTE_EST = 00
# DEPRECATED: the old single TIMEZONE=Australia/Sydney was a leftover from the
# ASX portfolio. It caused US/CANADA (cron 23:30/23:35 CST = ~01:30 Sydney) to
# fail the is_trading_time gate (hour<9) and silently never enter the rebalance
# block, while EUROPE (17:00 CST = ~19:00 Sydney) traded fine. Kept only as a
# fallback for any caller that still reads it; prefer get_market_timezone(region).
TIMEZONE = pytz.timezone("Australia/Sydney")

# ----------------------------------------------------------------------------
# Per-region market timezones (fix for the US/CANADA no-trade bug)
# ----------------------------------------------------------------------------
# AGENTS.md convention: "UTC for international, America/New_York for US stocks".
# main.py's daily-trading gate computes today_date + is_trading_time in the
# region's MARKET timezone so the "new trading day" / weekend checks use the
# correct local date. Cron fires ~1hr after each region's market open, so
# is_trading_time is True at launch for every region.
REGION_MARKET_TIMEZONE = {
    "US": "America/New_York",
    "EUROPE": "Europe/Berlin",
    "CANADA": "America/Toronto",
    "ASIA": "Asia/Tokyo",
    "MIDDLE_EAST": "Asia/Riyadh",
    "OCEANIA": "Australia/Sydney",
    "INDIA": "Asia/Kolkata",
    "ALL": "America/New_York",
}


def get_market_timezone(region: str):
    """Return the pytz timezone for a trading region's home market.

    Falls back to the legacy TIMEZONE (Sydney) for any region not in the map so
    callers never crash on an unexpected region string.
    """
    tz_name = REGION_MARKET_TIMEZONE.get(region)
    if tz_name is None:
        return TIMEZONE
    return pytz.timezone(tz_name)

# ============================================================================
# ORDER EXECUTION CONFIGURATION
# ============================================================================
#
# ORDER TYPE OPTIONS:
# - 'MARKET'   : Execute immediately at current market price (fastest fill, highest slippage)
# - 'LIMIT'    : Execute at specified price or better (price control, may not fill)
# - 'MIDPRICE' : IBKR's native midprice peg (pegged to NBBO midpoint, adaptive)
#
# LIMIT PRICE CALCULATION:
# - If ORDER_TYPE = 'LIMIT' and LIMIT_PRICE_OFFSET is set:
#     BUY:  limit_price = current_price + LIMIT_PRICE_OFFSET
#     SELL: limit_price = current_price - LIMIT_PRICE_OFFSET
# - If ORDER_TYPE = 'LIMIT' and LIMIT_PRICE_OFFSET is None/0: Falls back to MIDPRICE
#
# TIME-IN-FORCE: All orders use 'DAY' (expire at market close)
# ALL-OR-NONE:   Optional full-fill requirement (WARNING: not supported on all exchanges)
#
# ============================================================================

# Order type: 'MARKET', 'LIMIT', or 'MIDPRICE'
ORDER_TYPE = "MARKET"

# For LIMIT orders: Global price offset from current price
# Positive = more aggressive (willing to pay more for BUY, accept less for SELL)
# Negative = less aggressive (trying to get better price)
# None or 0 = use MIDPRICE behavior
# Examples:
#   0.05  = BUY at ask+$0.05, SELL at bid-$0.05 (aggressive, faster fill)
#   -0.05 = BUY at ask-$0.05, SELL at bid+$0.05 (passive, better price)
#   0     = Use midpoint of bid-ask (same as MIDPRICE)
LIMIT_PRICE_OFFSET = None  # None = use MIDPRICE if ORDER_TYPE is LIMIT

# For LIMIT orders: Percentage offset from current price (alternative to absolute offset)
# Example: 0.005 = 0.5% offset
#   BUY:  limit_price = ask * (1 + LIMIT_PRICE_OFFSET_PCT)  -> pay slightly more
#   SELL: limit_price = bid * (1 - LIMIT_PRICE_OFFSET_PCT)  -> accept slightly less
# Set to None to use LIMIT_PRICE_OFFSET (absolute) instead
# If both are set, percentage takes precedence
LIMIT_PRICE_OFFSET_PCT = 0.002  # 0.2% offset

# Intelligent LIMIT engine configuration (used when ORDER_TYPE == 'LIMIT')
# Strategy options:
# - 'CROSS_SPREAD': BUY at ask / SELL at bid (highest fill probability)
# - 'MIDPOINT': Use midpoint when bid and ask are available
# - 'OFFSET_FROM_NBBO': Apply LIMIT_PRICE_OFFSET(_PCT) over live IBKR NBBO
LIMIT_PRICE_STRATEGY = "CROSS_SPREAD"

# Wait time after each exchange market open before LIMIT submission
LIMIT_WAIT_AFTER_OPEN_MINUTES = 5
LIMIT_WAIT_AFTER_OPEN_BY_EXCHANGE = {
    # 'TSEJ': 15,
    # 'SMART': 5,
    # 'LSE': 10,
}

# Retry behavior for unfilled LIMIT orders
MAX_LIMIT_RETRIES = 3
LIMIT_FILL_TIMEOUT_SECONDS = 30
LIMIT_QUOTE_TIMEOUT_SECONDS = 10

# In-algo deferred order persistence file
PENDING_ORDERS_FILE = "pending_limit_orders.json"

# All-or-None: Require full fill (no partial fills)
# WARNING: AON is NOT supported on all exchanges:
#   - TSEJ (Tokyo): Does NOT support AON
#   - SEHK (Hong Kong): Does NOT support AON for some order types
#   - Most US exchanges: Support AON
# When enabled on unsupported exchanges, the order will be placed WITHOUT AON
ORDER_ALL_OR_NONE = False

# ============================================================================
# ORDER SUBMISSION TIMING
# ============================================================================
# Schedule when orders are submitted relative to each exchange's trading hours.
# This allows you to avoid volatile open/close periods.
#
# Options:
#   'IMMEDIATE'           - Submit as soon as signals are generated (default)
#   'AT_OPEN'             - Submit at market open
#   '30_MIN_AFTER_OPEN'   - Submit 30 minutes after market open
#   '1_HOUR_AFTER_OPEN'   - Submit 1 hour after market open
#   'MIDDAY'              - Submit at midday (varies by exchange)
#   '1_HOUR_BEFORE_CLOSE' - Submit 1 hour before market close
#   '30_MIN_BEFORE_CLOSE' - Submit 30 minutes before market close
#
# Note: Each exchange has different trading hours:
#   - SMART (US):    09:30-16:00 ET
#   - TSEJ (Tokyo):  09:00-15:00 JST (with 11:30-12:30 lunch break)
#   - LSE (London):  08:00-16:30 GMT
#   - SEHK (HK):     09:30-16:00 HKT (with 12:00-13:00 lunch break)
#   - BUX (Hungary): 09:00-17:30 CET
#   - SFB (Sweden):  09:00-17:30 CET
#   - BVME (Milan):  09:00-17:30 CET
#   - TADAWUL (Saudi): 10:00-15:00 AST (Sun-Thu)
#
# ============================================================================

# Default submission timing for all exchanges
ORDER_SUBMISSION_TIMING = "IMMEDIATE"

# Per-exchange timing overrides (optional)
# Uncomment and modify to set different timing per exchange
ORDER_SUBMISSION_TIMING_BY_EXCHANGE = {
    # 'SMART': 'IMMEDIATE',           # US - execute immediately
    # 'TSEJ': '30_MIN_AFTER_OPEN',    # Tokyo - avoid opening volatility
    # 'LSE': '1_HOUR_AFTER_OPEN',     # London - wait for market to settle
    # 'SEHK': 'MIDDAY',               # Hong Kong - after lunch break
    # 'BUX': 'MIDDAY',                # Budapest - midday
    # 'SFB': '30_MIN_AFTER_OPEN',     # Stockholm - after open
    # 'BVME': '30_MIN_AFTER_OPEN',    # Milan - after open
    # 'TADAWUL': '1_HOUR_AFTER_OPEN', # Saudi - after open
}

# ============================================================================
# REGION-BASED TRADING SESSIONS
# ============================================================================
# Use --region argument to run separate trading sessions for different regions.
# This allows running the script at appropriate times for each region's market.
#
# Example cron schedule for '1_HOUR_AFTER_OPEN':
#   04:00 EST -> python main.py --region EUROPE  (EU opens ~03:00-04:00 EST)
#   10:30 EST -> python main.py --region US      (US opens 09:30 EST)
#   20:00 EST -> python main.py --region ASIA    (Tokyo opens 19:00 EST = 09:00 JST)
#
# Or run all regions: python main.py --region ALL (default, processes all symbols)
# ============================================================================

REGION_EXCHANGES = {
    "US": ["SMART"],
    "EUROPE": [
        "LSE",  # London
        "SBF",  # Paris
        "IBIS",  # Frankfurt (Xetra)
        "FWB2",  # Frankfurt (Börse Frankfurt)
        "BUX",  # Budapest
        "SFB",  # Stockholm
        "BVME",  # Milan
        "AMS",  # Amsterdam
        "CPH",  # Copenhagen
        "WSE",  # Warsaw
        "EBR",  # Brussels
        "N.TALLINN",  # Tallinn
        "N.VILNIUS",  # Vilnius
        "VSE",  # Vienna
        "OSE",  # Oslo
        "SIX",  # Swiss
        "PRA",  # Prague
        "BM",  # Madrid
        "BVB",  # Bucharest
    ],
    "ASIA": [
        "TSEJ",  # Tokyo
        "SEHK",  # Hong Kong
        "SGX",  # Singapore
        "MYX",  # Malaysia
    ],
    "MIDDLE_EAST": [
        "TADAWUL",  # Saudi Arabia
        "DFM",  # Dubai
        "TASE",  # Tel Aviv
    ],
    "CANADA": ["TSE"],
    "OCEANIA": ["ASX"],
    "INDIA": ["NSE", "BSE"],
}

# ============================================================================
# LEGACY LIMIT ORDER SETTINGS (kept for backward compatibility)
# ============================================================================
# These are used when ORDER_TYPE is not set (legacy mode)
USE_LIMIT_ORDERS = False

# Default slippage tolerance for limit orders (0.5% = 50 basis points)
DEFAULT_LIMIT_SLIPPAGE_TOLERANCE = 0.005

# Per-exchange slippage tolerances (based on typical bid-ask spreads)
EXCHANGE_LIMIT_TOLERANCE = {
    "TSEJ": 0.0005,  # 0.05% - Tokyo Stock Exchange (tight spreads)
    "LSE": 0.0005,  # 0.05% - London Stock Exchange (moderate spreads)
    "NSE": 0.0005,  # 0.05% - India NSE (wider spreads, higher volatility)
    "BSE": 0.0005,  # 0.05% - India BSE
    "SEHK": 0.0005,  # 0.05% - Hong Kong
    "ASX": 0.0005,  # 0.05% - Australia
    "SBF": 0.0005,  # 0.05% - Euronext Paris
    "IBIS": 0.0005,  # 0.05% - Deutsche Börse
    "SMART": 0.0005,  # 0.05% - US markets (very liquid)
}

# ============================================================================
# CURRENCY RATE FALLBACKS
# ============================================================================
# Fallback currency rates used when IBKR and yfinance both fail.
# These are NOT overrides - live rates from IBKR/yfinance are always preferred.
# These rates are only used as a last resort to prevent trade failures.
#
# Format: XXX_TO_USD = rate (how many USD per 1 unit of XXX)
# Example: CAD_TO_USD = 0.72 means 1 CAD = 0.72 USD
#
# For currencies quoted as USD/XXX (JPY, HKD, HUF, etc.), use the inverse:
# Example: JPY_TO_USD = 0.00645 means 1 JPY = 0.00645 USD (i.e., USD/JPY ≈ 155)
#
# Update these periodically for accuracy, especially during high volatility.
# ============================================================================

CURRENCY_RATE_FALLBACKS = {
    # Major currencies (XXX/USD format - multiply)
    "CAD": 0.73397543,  # 1 CAD = 0.72 USD (CAD/USD)
    "GBP": 1.36933302,  # 1 GBP = 1.27 USD (GBP/USD)
    "EUR": 1.18709079,  # 1 EUR = 1.08 USD (EUR/USD)
    "AUD": 0.69638681,  # 1 AUD = 0.66 USD (AUD/USD)
    "NZD": 0.60265878,  # 1 NZD = 0.60 USD (NZD/USD)
    "CHF": 1.29471551,  # 1 CHF = 1.12 USD (CHF/USD)
    "SGD": 0.78661824,  # 1 SGD = 0.74 USD (SGD/USD)
    # Currencies quoted as USD/XXX (need inverse for XXX_TO_USD)
    # Formula: XXX_TO_USD = 1 / (USD/XXX rate)
    "JPY": 0.00645098,  # 1 JPY = 0.00645 USD (USD/JPY ≈ 155)
    "HKD": 0.12807554,  # 1 HKD = 0.128 USD (USD/HKD ≈ 7.80)
    "HUF": 0.00311066,  # 1 HUF = 0.00263 USD (USD/HUF ≈ 380)
    "SEK": 0.11226531,  # 1 SEK = 0.095 USD (USD/SEK ≈ 10.5)
    "CZK": 0.04872715,  # 1 CZK = 0.0426 USD (USD/CZK ≈ 23.5)
    "NOK": 0.10372524,  # 1 NOK = 0.0926 USD (USD/NOK ≈ 10.8)
    "PLN": 0.28161163,  # 1 PLN = 0.25 USD (USD/PLN ≈ 4.0)
    "DKK": 0.15895450,  # 1 DKK = 0.145 USD (USD/DKK ≈ 6.9)
    "ILS": 0.32259189,  # 1 ILS = 0.27 USD (USD/ILS ≈ 3.7)
    "MXN": 0.05724615,  # 1 MXN = 0.059 USD (USD/MXN ≈ 17.0)
    "ZAR": 0.06183810,  # 1 ZAR = 0.054 USD (USD/ZAR ≈ 18.5)
    "TRY": 0.02298438,  # 1 TRY = 0.031 USD (USD/TRY ≈ 32.0)
    "RON": 0.21700000,  # 1 RON = 0.217 USD (USD/RON ≈ 4.6)
    "INR": 0.01087731,  # 1 INR = 0.012 USD (USD/INR ≈ 83.0)
    # Pegged currencies (fixed rate)
    "SAR": 0.26666666,  # 1 SAR = 0.267 USD (USD/SAR = 3.75, pegged)
}

# ============================================================================
# KILL-SWITCH (Phase 0 of Revision Protocol — see docs/REVISION_POLICY.md)
# ============================================================================
# Thresholds are PLACEHOLDERS. Phase 2 will replace them with values
# derived from backtest distribution percentiles.
#
# Sentinel files (created by kill_switch.py) MUST be deleted manually
# to re-enable trading:
#   execution/KILL_SWITCH_ACTIVE   (hard kill)
#   execution/SOFT_HALT_ACTIVE     (soft halt)
#   execution/DAILY_MOVE_ACTIVE    (daily-move alarm)

KILL_SWITCH_HARD_DD = 0.08         # MTD drawdown ≥ 8% → hard kill
KILL_SWITCH_SOFT_DD = 0.05         # MTD drawdown ≥ 5% → soft halt
KILL_SWITCH_DAILY_MOVE = 0.04      # Daily move ≥ ±4% → daily-move alarm
KILL_SWITCH_RETAIN_TICKERS = ["BIL", "TLT"]  # Not flattened on hard kill (example)
KILL_SWITCH_ENABLED = True         # Master switch; set False ONLY in emergencies

# File Paths
LOG_FILE = "algo_trading.log"
REPORT_FILE = "trading_report.log"

# Logging Configuration
LOG_LEVEL = logging.INFO
