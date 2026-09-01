import time
import random
import zmq
import sys
import os
import datetime as dt
import pytz
import traceback
import argparse
from pathlib import Path as _Path

# Load .env into os.environ (only keys not already set). Cron does not inherit
# the interactive shell's exports; this gives the live run its Telegram/IBKR
# credentials. See execution/env_loader.py.
sys.path.insert(0, str(_Path(__file__).resolve().parent))
try:
    import env_loader  # noqa: F401  (side-effect: populates os.environ)
except Exception:
    pass

from market_calendar import MarketCalendarManager  # Per-exchange holiday detection

from ibapi.contract import Contract
from ibapi.order import Order

from config import (
    LOG_FILE,
    LOG_LEVEL,
    SYMBOLS,
    ASSET_SPECIFIC_CONFIGS,
    REPORT_FILE,
    TRADING_HOUR_EST,
    TRADING_MINUTE_EST,
    TIMEZONE,
    get_market_timezone,
    OVERRIDE_ML_FOR_INITIAL_ENTRY,
    BLACKLISTED_SYMBOLS,
    ORDER_TYPE,
    LIMIT_PRICE_OFFSET,
    LIMIT_PRICE_OFFSET_PCT,
    ORDER_ALL_OR_NONE,
    ORDER_SUBMISSION_TIMING,
    ORDER_SUBMISSION_TIMING_BY_EXCHANGE,
    REGION_EXCHANGES,
    LIMIT_PRICE_STRATEGY,
    LIMIT_WAIT_AFTER_OPEN_MINUTES,
    LIMIT_WAIT_AFTER_OPEN_BY_EXCHANGE,
    MAX_LIMIT_RETRIES,
    LIMIT_FILL_TIMEOUT_SECONDS,
    LIMIT_QUOTE_TIMEOUT_SECONDS,
    PENDING_ORDERS_FILE,
    STRATEGY_MODE,
    WEEKLY_GATE_CONFIG,
    HRP_BASE_WEIGHTS,
    allocate_session_client_id,
    release_session_client_id,
)
from utils import (
    setup_logger,
    setup_zmq_publisher,
    generate_report,
    get_current_time_in_timezone,
)

# Use final clean IB client
from ib_client_final import IBClient
from data_manager import DataManager
from portfolio_manager import PortfolioManager
from order_manager import OrderManager
from strategy_executor import StrategyExecutor
from exchange_manager import ExchangeManager
from forex_manager import ForexManager
from config import CASH_REBALANCING_MODE

# Revision Protocol — Phase 0 kill-switch wiring.
# See docs/superpowers/plans/REVISION_PROTOCOL_PLAN.md
from kill_switch_runtime import (
    record_equity_event,
    evaluate_and_apply,
    is_hard_kill_active,
    is_soft_halt_active,
)

# Conditionally import CashPortfolioManager for phase1/phase2 modes
_cash_pm_available = False
if CASH_REBALANCING_MODE in ("phase1", "phase2"):
    try:
        from cash_portfolio_manager import CashPortfolioManager

        _cash_pm_available = True
    except ImportError as _e:
        print(
            f"Warning: CashPortfolioManager not available ({_e}). Falling back to legacy mode."
        )
        CASH_REBALANCING_MODE = "legacy"
from contract_details_manager import ContractDetailsManager
from currency_converter import CurrencyConverter
from order_scheduler import OrderScheduler
from market_data_manager import MarketDataManager
from pending_order_book import PendingOrderBook
from limit_order_engine import LimitOrderEngine
# Connection monitor removed - assuming stable connection

# Initialize exchange manager for multi-exchange support (used before connection)
exchange_manager = None


def filter_symbols_by_region(
    symbols: list, region: str, exchange_manager_instance
) -> list:
    """
    Filter symbols to only include those from the specified region.

    Args:
        symbols: List of symbols in yfinance format (e.g., ['NVDA', '8002.T', 'III.L'])
        region: Region name ('US', 'EUROPE', 'ASIA', etc.) or 'ALL' for no filtering
        exchange_manager_instance: ExchangeManager instance for symbol->exchange mapping

    Returns:
        Filtered list of symbols belonging to the specified region
    """
    if region == "ALL":
        return symbols

    # Get exchanges for this region
    region_upper = region.upper()
    if region_upper not in REGION_EXCHANGES:
        valid_regions = list(REGION_EXCHANGES.keys()) + ["ALL"]
        raise ValueError(f"Unknown region '{region}'. Valid regions: {valid_regions}")

    target_exchanges = set(REGION_EXCHANGES[region_upper])

    # Filter symbols
    filtered = []
    for symbol in symbols:
        exchange = exchange_manager_instance.get_exchange(symbol)
        if exchange in target_exchanges:
            filtered.append(symbol)

    return filtered


def create_contract(symbol):
    """
    Helper to create an IB Contract object with multi-exchange support.

    Supports:
    - Tokyo Stock Exchange: 8002.T -> 8002 on TSE (JPY)
    - London Stock Exchange: III.L -> III on LSE (GBP)
    - US Markets: NVDA -> NVDA on SMART (USD)

    Args:
        symbol: Symbol in yfinance format (e.g., '8002.T', 'III.L', 'NVDA')

    Returns:
        ib_insync Contract object configured for the appropriate exchange
    """
    return exchange_manager.create_contract(symbol)


def wait_for_orders_to_fill(order_manager, logger, timeout=300, poll_interval=10):
    """
    Smart order fill monitoring - polls order status and proceeds when all filled.

    Args:
        order_manager: OrderManager instance
        logger: Logger instance
        timeout: Maximum wait time in seconds (default 300 = 5 minutes)
        poll_interval: Check interval in seconds (default 10s)

    Returns:
        bool: True if all orders filled, False if timeout reached
    """
    start_time = time.time()
    last_status_time = start_time

    logger.info(f"\n{'=' * 60}")
    logger.info("MONITORING ORDER FILLS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Timeout: {timeout}s | Poll interval: {poll_interval}s")

    while time.time() - start_time < timeout:
        open_orders = order_manager.get_open_orders()

        # Check if all orders are filled
        if not open_orders:
            elapsed = time.time() - start_time
            logger.info(f"\n✅ All orders filled in {elapsed:.1f} seconds")
            logger.info(f"{'=' * 60}")
            return True

        # Log status every poll_interval
        elapsed = time.time() - start_time
        logger.info(
            f"\n[{elapsed:.1f}s] Waiting for {len(open_orders)} orders to fill..."
        )

        for order_id, info in open_orders.items():
            symbol = info["contract"].symbol
            status = info["status"]
            filled = info.get("filledQuantity", 0)
            total = info.get("totalQuantity", 0)
            action = info.get("action", "")

            logger.info(
                f"  Order {order_id} ({symbol}): {status} | "
                f"{action} {filled}/{total} filled"
            )

        # Sleep until next poll
        time.sleep(poll_interval)

    # Timeout reached
    elapsed = time.time() - start_time
    open_orders = order_manager.get_open_orders()

    logger.warning(f"\n⏰ Timeout reached after {elapsed:.1f}s")
    logger.warning(f"{len(open_orders)} orders still pending:")

    for order_id, info in open_orders.items():
        symbol = info["contract"].symbol
        status = info["status"]
        logger.warning(f"  Order {order_id} ({symbol}): {status}")

    logger.info(f"{'=' * 60}")
    return False


# Per-region ZeroMQ publisher ports. Sessions overlap (EUROPE 17:00 can run
# until ~23:45, past US 23:30 / CANADA 23:35), so a single shared port caused
# bind() collisions that crashed main.py before trading. Distinct ports per
# region remove the collision; ALL keeps the historical default.
ZMQ_REGION_PORTS = {
    "US": 5555,
    "EUROPE": 5556,
    "CANADA": 5557,
    "ASIA": 5558,
    "MIDDLE_EAST": 5559,
    "OCEANIA": 5560,
    "INDIA": 5561,
    "ALL": 5555,
}


def main(region: str = "ALL"):
    # Ensure strategy_models directory exists for dummy algos
    os.makedirs("strategy_models", exist_ok=True)

    # 1. Setup ZeroMQ Publisher for external monitoring (per-region port).
    # The publisher is monitoring-only: if the port is still held by an
    # overlapping session we log and continue WITHOUT it rather than aborting
    # the trading session (a monitoring socket must never block trading).
    zmq_port = ZMQ_REGION_PORTS.get(region, 5555)
    try:
        zmq_context, zmq_socket = setup_zmq_publisher(port=zmq_port)
    except zmq.ZMQError as e:
        zmq_context, zmq_socket = None, None
        print(
            f"WARNING: ZMQ publisher unavailable on port {zmq_port} for region "
            f"{region} ({e}). Continuing WITHOUT external monitoring; trading "
            f"proceeds normally."
        )

    # 2. Setup Logger
    logger = setup_logger(LOG_FILE, level=LOG_LEVEL, zmq_pub_socket=zmq_socket)
    logger.info("Starting algo trading system...")
    logger.info(f"Region filter: {region}")
    if zmq_socket is None:
        logger.warning(
            f"ZMQ publisher disabled (port {zmq_port} unavailable). External "
            f"monitoring feed is OFF for this session; trading is unaffected."
        )

    # 3. Initialize Exchange Manager (needed before creating contracts)
    global exchange_manager
    exchange_manager = ExchangeManager(logger)

    # 3a. Filter symbols by region (must be after exchange_manager initialization)
    active_symbols = filter_symbols_by_region(SYMBOLS, region, exchange_manager)

    if not active_symbols:
        logger.warning(
            f"No symbols found for region '{region}'. Available symbols by region:"
        )
        for r, exchanges in REGION_EXCHANGES.items():
            r_symbols = filter_symbols_by_region(SYMBOLS, r, exchange_manager)
            if r_symbols:
                logger.warning(f"  {r}: {r_symbols}")
        logger.warning("Exiting - no symbols to trade.")
        sys.exit(0)

    logger.info(
        f"Trading {len(active_symbols)} symbols for region '{region}': {active_symbols}"
    )

    # CRITICAL: Pre-trade symbol validation
    # This catches unmapped suffixes BEFORE any trading happens
    logger.info("\n" + "=" * 70)
    logger.info("PRE-TRADE SYMBOL VALIDATION (CRITICAL)")
    logger.info("=" * 70)

    # Check for unmapped suffixes first (catches config errors)
    unmapped = exchange_manager.get_unmapped_suffixes()
    if unmapped:
        logger.critical("❌ UNMAPPED EXCHANGE SUFFIXES DETECTED!")
        logger.critical("The following symbols have suffixes not in EXCHANGE_SUFFIXES:")
        for symbol, suffix in unmapped:
            logger.critical(
                f"  {symbol} has suffix '{suffix}' -> would incorrectly route to SMART (US)!"
            )
        logger.critical("")
        logger.critical(
            "FIX: Add suffix to EXCHANGE_SUFFIXES or add symbol to SYMBOL_OVERRIDES in exchange_manager.py"
        )
        logger.critical("REFUSING TO START - fix config and restart.")
        sys.exit(1)

    # Full symbol validation
    validation_results = exchange_manager.validate_config_symbols()
    invalid_symbols = [s for s, r in validation_results.items() if not r["valid"]]
    if invalid_symbols:
        logger.critical(f"❌ INVALID SYMBOLS: {invalid_symbols}")
        logger.critical("REFUSING TO START - fix config and restart.")
        sys.exit(1)

    logger.info("✅ All symbols validated - safe to proceed with trading\n")

    # 4. Initialize Managers and Client
    data_manager = DataManager(
        logger, exchange_manager=exchange_manager
    )  # Pass exchange_manager for symbol conversion
    order_manager = OrderManager(logger)
    # Set lags to 5 in strategy_executor to match data_manager's default
    strategy_executor = StrategyExecutor(data_manager, logger, lags=5)

    # Initialize weekly gate manager if in weekly_gated mode
    weekly_gate_mgr = None
    if STRATEGY_MODE == "weekly_gated":
        from weekly_gate_manager import WeeklyGateManager

        weekly_gate_mgr = WeeklyGateManager(
            hrp_base_weights=HRP_BASE_WEIGHTS,
            strategy_executor=strategy_executor,
            config=WEEKLY_GATE_CONFIG,
        )
        logger.info(
            f"[Main] Strategy mode: WEEKLY_GATED ({len(HRP_BASE_WEIGHTS)} tickers)"
        )
    else:
        logger.info("[Main] Strategy mode: DAILY_BINARY")

    # Initialize PortfolioManager without multi-currency support yet (will be added after connection)
    portfolio_manager = PortfolioManager(logger)

    # Unique, collision-checked IBKR client ID from the shared rotator so
    # overlapping sessions (EUROPE running past US/CANADA cron times, plus data
    # downloads / oversight) never collide on a shared ID (IBKR error 326). The
    # rotator is registry- + live-probe-aware; it falls back to the legacy
    # per-region static ID only if it cannot be imported. Released on shutdown.
    region_client_id = allocate_session_client_id(region)
    logger.info(
        f"Using IBKR API client ID {region_client_id} for region '{region}' "
        f"(allocated via client-id rotator)."
    )
    ib_client = IBClient(
        portfolio_manager, order_manager, logger, client_id=region_client_id
    )

    # Initialize forex manager for JPY carry trade (will be called after stock trades)
    # Pass exchange_manager for intelligent currency detection from config
    forex_manager = ForexManager(ib_client, logger, exchange_manager=exchange_manager)

    # Initialize cash portfolio manager (phase1/phase2 modes)
    cash_portfolio_manager = None
    if CASH_REBALANCING_MODE in ("phase1", "phase2") and _cash_pm_available:
        cash_portfolio_manager = CashPortfolioManager(
            ib_client=ib_client,
            logger=logger,
            strategy_executor=strategy_executor
            if CASH_REBALANCING_MODE == "phase2"
            else None,
            exchange_manager=exchange_manager,
            currency_converter=None,  # Wired after IBKR connection
            dry_run=False,
        )
        logger.info(
            f"Cash portfolio manager initialized (mode: {CASH_REBALANCING_MODE})"
        )

    # 5. Connect to IBKR (PRODUCTION ONLY - NO OFFLINE MODE)
    if not ib_client.connect_and_run():
        logger.critical("Failed to connect to IB Gateway. Cannot proceed.")
        logger.critical(
            "Please follow the instructions above to resolve the connection issue."
        )
        sys.exit(1)

    logger.info("Connected to IBKR successfully in PRODUCTION mode")

    # 5a. Discover pre-existing open orders from IBKR
    # This populates order_manager with orders from prior sessions or TWS manual orders.
    # Must happen BEFORE any trading logic runs.
    logger.info("=" * 60)
    logger.info("CHECKING FOR PRE-EXISTING OPEN ORDERS")
    logger.info("=" * 60)
    ib_client.request_all_open_orders(timeout=5.0)
    preexisting = order_manager.get_open_orders()
    if preexisting:
        logger.warning(f"Found {len(preexisting)} pre-existing open order(s):")
        for oid, info in preexisting.items():
            logger.warning(
                f"  Order {oid}: {info['action']} "
                f"{info['totalQuantity']} {info['contract'].symbol} "
                f"- {info['status']}"
            )
    else:
        logger.info("No pre-existing open orders found.")

    # 6. Initialize Multi-Currency Support (after IB connection established)
    logger.info("Initializing multi-currency support managers...")

    # Create ContractDetailsManager to fetch priceMagnifier from IBKR
    contract_details_mgr = ContractDetailsManager(ib_client, logger)
    ib_client.contract_details_mgr = contract_details_mgr  # Attach for callbacks

    # Create CurrencyConverter to fetch forex rates from IBKR
    currency_converter = CurrencyConverter(ib_client, logger)
    ib_client.currency_converter = currency_converter  # Attach for callbacks

    # Create MarketDataManager for live stock bid/ask snapshots from IBKR
    market_data_mgr = MarketDataManager(ib_client, logger)
    ib_client.market_data_mgr = market_data_mgr

    # Create deferred order storage and intelligent LIMIT execution engine
    pending_order_book = PendingOrderBook(logger, PENDING_ORDERS_FILE)
    pending_order_book.load()

    limit_order_engine = LimitOrderEngine(
        ib_client,
        order_manager,
        market_data_mgr,
        exchange_manager,
        contract_details_mgr,
        logger,
    )

    # Initialize pre-flight order validation system
    from execution_journal import ExecutionJournal
    from order_guard import OrderGuard

    journal = ExecutionJournal(
        journal_dir="execution_journals", logger=logger, retention_days=30
    )
    order_manager.journal = journal  # Hook fill events into journal

    order_guard = OrderGuard(
        ib_client=ib_client,
        order_manager=order_manager,
        journal=journal,
        logger=logger,
    )
    logger.info("Pre-flight order validation system initialized (OrderGuard + Journal)")

    # Initialize IBKR Data Manager (primary data source for historical + price data)
    from ibkr_data_manager import IBKRDataManager

    ibkr_data_mgr = IBKRDataManager(
        ib_client=ib_client,
        contract_details_mgr=contract_details_mgr,
        exchange_manager=exchange_manager,
        logger=logger,
    )
    logger.info("IBKR Data Manager initialized (primary data source)")

    # Wire managers to portfolio_manager
    portfolio_manager.exchange_manager = exchange_manager
    portfolio_manager.contract_details_mgr = contract_details_mgr
    portfolio_manager.currency_converter = currency_converter

    # Wire IBKR data manager as primary data source
    data_manager.ibkr_data_manager = ibkr_data_mgr
    portfolio_manager.ibkr_data_manager = ibkr_data_mgr
    currency_converter.ibkr_data_manager = ibkr_data_mgr

    portfolio_manager.data_manager = data_manager
    portfolio_manager.market_data_manager = market_data_mgr

    # Wire post-connection managers to cash_portfolio_manager
    if CASH_REBALANCING_MODE in ("phase1", "phase2") and cash_portfolio_manager:
        cash_portfolio_manager.currency_converter = currency_converter
        cash_portfolio_manager.order_guard = order_guard
        cash_portfolio_manager.ibkr_data_manager = ibkr_data_mgr

    # 7. Fetch contract details for all symbols AND VERIFY against our mappings
    # This is the SECOND layer of protection - IBKR confirms our mappings are correct
    logger.info("\n" + "=" * 70)
    logger.info("IBKR CONTRACT VERIFICATION (SECOND LAYER)")
    logger.info("=" * 70)
    logger.info(
        "Fetching contract details from IBKR and verifying against our mappings..."
    )

    contract_mismatches = []

    for symbol in active_symbols:
        contract = create_contract(symbol)
        details = contract_details_mgr.fetch_contract_details(
            symbol, contract, timeout=10.0
        )

        # Get our expected values
        expected_ibkr, expected_exchange, expected_currency = (
            exchange_manager.parse_symbol(symbol)
        )

        if details:
            ibkr_currency = details.get("currency", "UNKNOWN")
            ibkr_exchange = details.get("exchange", "UNKNOWN")

            # Verify currency matches
            if ibkr_currency != expected_currency:
                mismatch_msg = f"CURRENCY MISMATCH: {symbol} - expected {expected_currency}, IBKR returned {ibkr_currency}"
                contract_mismatches.append(mismatch_msg)
                logger.error(f"✗ {symbol}: {mismatch_msg}")
            else:
                logger.info(
                    f"✓ {symbol}: priceMagnifier={details.get('priceMagnifier', 1)}, "
                    f"currency={ibkr_currency} (verified ✓), exchange={ibkr_exchange}"
                )
        else:
            logger.warning(
                f"⚠ {symbol}: Could not fetch contract details from IBKR - VERIFY MANUALLY!"
            )

    logger.info("-" * 70)

    if contract_mismatches:
        logger.critical("❌ CONTRACT MISMATCHES DETECTED!")
        for msg in contract_mismatches:
            logger.critical(f"  {msg}")
        logger.critical("")
        logger.critical(
            "This means our exchange_manager.py mappings don't match what IBKR expects!"
        )
        logger.critical("REFUSING TO START - fix mappings and restart.")
        sys.exit(1)

    logger.info("✅ All contracts verified with IBKR - mappings are correct!")
    logger.info("=" * 70 + "\n")

    # 8. Request next valid order ID
    if not ib_client.request_next_valid_id():
        logger.critical("Could not get a valid order ID. Exiting.")
        ib_client.disconnect_ib()
        sys.exit(1)

    # 9. Request account updates (for Net Liquidation and account values)
    logger.info("Requesting account updates...")
    ib_client.request_account_updates()
    time.sleep(2)  # Brief wait for account values

    # 10. CRITICAL: Request positions with event-driven wait
    # This prevents the fatal bug where system trades without knowing current positions
    logger.info("=" * 60)
    logger.info("LOADING CURRENT POSITIONS (CRITICAL SAFETY CHECK)")
    logger.info("=" * 60)
    logger.info("Expected positions:")
    logger.info("  HSBC: 922 shares")
    logger.info("  XOM: 853 shares")
    logger.info("  AVGO: 1053 shares")
    logger.info("  WELL: 1222 shares")
    logger.info("  UGL: 2743 shares")
    logger.info("  8002.T: 7200 shares")
    logger.info("")

    if not ib_client.request_positions(timeout=30):
        logger.critical("=" * 60)
        logger.critical("FATAL ERROR: Failed to load positions within 30 seconds")
        logger.critical("Cannot proceed - would trade on incomplete/missing data!")
        logger.critical("This would cause DUPLICATE BUY ORDERS every restart!")
        logger.critical("=" * 60)
        ib_client.disconnect_ib()
        sys.exit(1)

    # 11. Verify net liquidation received
    if not portfolio_manager.get_current_net_liquidation():
        logger.critical("FATAL: Net Liquidation not available after loading positions")
        logger.critical("Cannot calculate position sizes without Net Liquidation")
        ib_client.disconnect_ib()
        sys.exit(1)

    # 12. Verify we have positions (or confirm starting fresh)
    position_count = len(portfolio_manager.current_positions)
    logger.info("=" * 60)
    logger.info(f"POSITION LOADING COMPLETE: {position_count} positions loaded")
    logger.info("=" * 60)

    if position_count == 0:
        logger.warning(
            "Starting with ZERO positions (new account or all positions closed)"
        )
        logger.warning("System will place initial BUY orders for all allocations")
    else:
        logger.info(f"Current positions verified:")
        for symbol, pos_data in portfolio_manager.current_positions.items():
            position = float(pos_data.get("position", 0))
            if position != 0:
                logger.info(f"  {symbol}: {position} shares")

    logger.info("=" * 60)

    # CRITICAL: Request per-currency cash balances for JPY carry trade
    # reqAccountUpdates() only returns USD - need reqAccountSummary for GBP, JPY, EUR
    # Use forex_manager to get expected currencies from config (not just existing positions)
    logger.info("=" * 60)
    logger.info("LOADING PER-CURRENCY CASH BALANCES (for carry trade)")
    logger.info("=" * 60)

    # Get expected currencies from ALL sources:
    # 1. Stock positions (from exchange_manager / SYMBOLS suffix detection)
    # 2. Carry trade exotic_routing config (covers currencies from old positions)
    # 3. Major currencies + carry target from CASH_PORTFOLIO_CONFIG
    expected_currencies = set(forex_manager.get_expected_currencies_from_config())
    if CASH_REBALANCING_MODE in ("phase1", "phase2"):
        from config import CASH_PORTFOLIO_CONFIG

        expected_currencies.update(
            CASH_PORTFOLIO_CONFIG.get("exotic_routing", {}).keys()
        )
        expected_currencies.add(CASH_PORTFOLIO_CONFIG.get("funding_currency", "USD"))
        expected_currencies.add(CASH_PORTFOLIO_CONFIG.get("carry_currency", "JPY"))
    expected_currencies = sorted(expected_currencies)
    logger.info(f"Requesting balances for: {expected_currencies}")

    # Request balances for ALL expected currencies
    currency_balances = ib_client.request_currency_balances(
        currencies=expected_currencies
    )

    if currency_balances:
        logger.info("Per-currency cash balances:")
        for curr, balance in sorted(currency_balances.items()):
            status = "DEBT" if balance < 0 else "CREDIT" if balance > 0 else "ZERO"
            logger.info(f"  {curr}: {balance:>15,.2f} ({status})")
    else:
        logger.warning(
            "No currency-specific balances loaded - carry trade may not detect non-USD debts"
        )

    logger.info("=" * 60)

    # Initialize subscribed_contracts and populate it with Contract objects
    subscribed_contracts = {}
    for symbol in active_symbols:
        contract = create_contract(symbol)
        subscribed_contracts[symbol] = contract

    # Live stock quote snapshots are requested on demand from IBKR for LIMIT orders.
    # Historical daily bars remain sourced from yfinance.

    # Initialize market calendar for per-exchange holiday detection
    market_calendar = MarketCalendarManager(logger)

    # Main Trading Loop - Now adapted for daily processing
    last_trading_day_processed = None
    report_interval = 300
    last_report_time = time.time()
    daily_check_interval = 60

    logger.info("Starting daily trading loop.")
    logger.info("--------------------------------------------------")

    try:
        while ib_client.isConnected():
            # Use the REGION's market timezone (not the legacy Sydney TIMEZONE)
            # so today_date and the is_trading_time gate reflect the market the
            # session is trading. The old single Sydney timezone caused US/CANADA
            # (cron ~01:30 Sydney) to never enter the rebalance block.
            region_tz = get_market_timezone(region)
            current_local_time = get_current_time_in_timezone(region_tz)
            today_date = current_local_time.date()

            # Check if it's a new day and past the trigger time. Cron fires ~1hr
            # after each region's market open, so this is True at launch for all
            # regions when using the correct market timezone.
            is_trading_time = current_local_time.hour > TRADING_HOUR_EST or (
                current_local_time.hour == TRADING_HOUR_EST
                and current_local_time.minute >= TRADING_MINUTE_EST
            )

            if today_date != last_trading_day_processed and is_trading_time:
                logger.info(
                    f"It's a new trading day ({today_date}) and past the daily trigger time ({TRADING_HOUR_EST}:{TRADING_MINUTE_EST} EST). Initiating daily trading logic check."
                )

                # Set last_trading_day_processed to today to prevent multiple runs
                last_trading_day_processed = today_date

                # Reset OrderGuard for new trading day
                order_guard.reset_trading_day()

                # ------------------------------------------------------------
                # Revision Protocol — Phase 0 kill-switch gate (start of day)
                # ------------------------------------------------------------
                # Tag the strategy executor with the current region so the
                # signal_history writer (Phase 1.1) attributes rows correctly.
                strategy_executor.current_region = region

                # 1. Record start-of-day equity event.
                # 2. Evaluate kill-switch; if hard kill, exit cleanly.
                record_equity_event(
                    portfolio_manager,
                    region=region,
                    event="start",
                    logger=logger,
                )
                _decision = evaluate_and_apply(
                    portfolio_manager,
                    region=region,
                    logger=logger,
                )
                if is_hard_kill_active():
                    logger.error(
                        "HARD KILL active. Exiting main.py. Remove "
                        "execution/KILL_SWITCH_ACTIVE to re-enable."
                    )
                    return
                if is_soft_halt_active():
                    logger.warning(
                        "SOFT HALT active. Flipping ml_signal tickers to "
                        "buy_and_hold for THIS PROCESS only (mirrors KILL_SWITCH.md "
                        "manual procedure). Restart with sentinel removed to revert."
                    )
                    _flipped = 0
                    for _sym, _cfg in ASSET_SPECIFIC_CONFIGS.items():
                        if _cfg.get("strategy_type") == "ml_signal":
                            _cfg["strategy_type"] = "buy_and_hold"
                            _flipped += 1
                    logger.warning(
                        "SOFT HALT: flipped %d ml_signal -> buy_and_hold", _flipped
                    )

                # --- Check for Weekends ---
                if today_date.weekday() in [5, 6]:  # 5 is Saturday, 6 is Sunday
                    logger.info(
                        f"Skipping daily trading logic for {today_date}: It's a weekend."
                    )
                    time.sleep(daily_check_interval)
                    continue

                # --- Per-Exchange Market Status Check ---
                # Filter symbols based on which exchanges are open today
                tradeable_symbols, closed_symbols = (
                    market_calendar.filter_tradeable_symbols(
                        active_symbols, today_date, exchange_manager
                    )
                )

                # Log market status summary
                logger.info(
                    market_calendar.get_market_status_summary(
                        active_symbols, today_date, exchange_manager
                    )
                )

                if not tradeable_symbols:
                    logger.info(
                        f"All exchanges closed on {today_date}. Skipping trading."
                    )
                    time.sleep(daily_check_interval)
                    continue

                if closed_symbols:
                    logger.info(
                        f"Skipping {len(closed_symbols)} symbols (market closed): {closed_symbols}"
                    )

                logger.info(
                    f"Proceeding with {len(tradeable_symbols)} tradeable symbols: {tradeable_symbols}"
                )

                # --- 8. Data Acquisition with INTELLIGENT market-aware date selection ---
                # Fetch latest available close for each TRADEABLE symbol based on market hours
                logger.info(
                    f"Fetching latest daily data for {len(tradeable_symbols)} tradeable symbols..."
                )

                for symbol in tradeable_symbols:
                    # Get exchange for this symbol to determine last trading day
                    _, symbol_exchange, _ = exchange_manager.parse_symbol(symbol)

                    # Get the last trading day for this specific exchange
                    # This handles exchange-specific holidays correctly
                    target_date = market_calendar.get_last_trading_date(
                        symbol_exchange, today_date
                    )
                    yfinance_end = today_date  # yfinance end is exclusive

                    logger.info(
                        f"{symbol} ({symbol_exchange}): Using {target_date} close"
                    )

                    # Fetch data
                    data_manager.fetch_and_store_historical_data(symbol, yfinance_end)

                    # No inter-symbol delay needed — IBKRDataManager handles pacing internally.
                    # yfinance fallback (if triggered) has its own rate limiting.

                # Give some time for data_manager to process
                time.sleep(2)

                # --- 9. Strategy Execution & Signal Generation ---
                signals = {}

                if STRATEGY_MODE == "weekly_gated" and weekly_gate_mgr is not None:
                    # Weekly gated mode: get gated allocation, override target weights
                    gated_alloc = weekly_gate_mgr.get_gated_allocation(
                        today_date, tradeable_symbols
                    )
                    portfolio_manager.target_allocation = gated_alloc
                    # Gated-in tickers get signal +1 (long), gated-out get -1 (sell)
                    for symbol in tradeable_symbols:
                        signals[symbol] = 1 if gated_alloc.get(symbol, 0) > 0 else -1
                else:
                    # Daily binary mode: existing per-ticker signal generation
                    for symbol in tradeable_symbols:
                        # Check strategy type - skip ML for buy-and-hold tickers
                        asset_config = ASSET_SPECIFIC_CONFIGS.get(symbol, {})
                        strategy_type = asset_config.get("strategy_type", "ml_signal")

                        if strategy_type == "buy_and_hold":
                            signals[symbol] = 1
                            logger.info(
                                f"{symbol}: Buy-and-hold strategy - signal forced to +1"
                            )
                            continue

                        # ML-based strategy: Generate signal from model
                        if data_manager.get_latest_historical_bar(symbol) is not None:
                            try:
                                signal = strategy_executor.generate_signal(symbol)
                                signals[symbol] = signal
                            except RuntimeError as e:
                                logger.error(
                                    f"CRITICAL: Signal generation failed for {symbol}: {e}"
                                )
                                logger.error(
                                    f"Skipping {symbol} for this trading cycle. Check data and model."
                                )
                                continue
                        else:
                            logger.error(
                                f"CRITICAL: No historical daily data available for {symbol}."
                            )
                            logger.error(
                                f"Check data fetching for {symbol}. Data fetch status: "
                                f"{data_manager.get_latest_historical_bar(symbol)}"
                            )
                            continue

                # --- 10. Signal-Based Trading with Position Rebalancing ---
                # This now handles:
                # 1. Closing unwanted/blacklisted positions
                # 2. Checking available capital
                # 3. Executing signal-based trades with proper position management
                trades_to_execute = (
                    portfolio_manager.get_trades_for_signal_based_execution(
                        signals, tradeable_symbols
                    )
                )

                # === CROSS-MARKET TRANSITION SAFETY CHECK ===
                # Prevents over-leverage when trading across multiple exchanges
                # with partial market closures (e.g., MLK Day: US closed, EU open)
                from config import TRANSITION_SAFETY_ENABLED

                if trades_to_execute and TRANSITION_SAFETY_ENABLED:
                    from transition_safety import TransitionSafetyManager, RiskLevel
                    from user_confirmation import (
                        UserConfirmationManager,
                        TransitionAction,
                    )

                    logger.info("\n" + "=" * 70)
                    logger.info("CROSS-MARKET TRANSITION SAFETY CHECK")
                    logger.info("=" * 70)

                    safety_mgr = TransitionSafetyManager(
                        portfolio_manager, market_calendar, exchange_manager, logger
                    )

                    metrics = safety_mgr.analyze_transition(
                        today_date, trades_to_execute, signals
                    )
                    report = safety_mgr.generate_report(metrics)
                    logger.info(report)

                    # HARD BLOCK: Critical risk level (exceeds margin or critical leverage)
                    if safety_mgr.should_block_execution(metrics):
                        logger.critical("BLOCKED: Trades exceed safety constraints!")

                        if metrics.exceeds_excess_liquidity:
                            logger.critical(
                                f"  Buy orders (${metrics.total_buy_value:,.0f}) exceed "
                                f"margin capacity (${metrics.max_allowed_buy_value:,.0f})"
                            )

                        if metrics.exceeds_critical_leverage:
                            logger.critical(
                                f"  Post-trade leverage ({metrics.post_trade_leverage:.2f}x) exceeds "
                                f"critical limit ({safety_mgr.leverage_critical}x)"
                            )

                        # Option: Scale down to fit
                        scaled_trades = safety_mgr.get_safe_trade_subset(
                            trades_to_execute, metrics
                        )
                        if scaled_trades:
                            logger.warning(
                                f"Auto-scaling down trades to fit constraints"
                            )
                            logger.warning(f"  Original: {trades_to_execute}")
                            logger.warning(f"  Scaled:   {scaled_trades}")
                            trades_to_execute = scaled_trades
                        else:
                            logger.warning(
                                "Cannot scale down - cancelling all BUY orders"
                            )
                            trades_to_execute = safety_mgr.get_sells_only(
                                trades_to_execute
                            )
                            if not trades_to_execute:
                                logger.info(
                                    "No SELL orders to execute. Skipping trading today."
                                )

                    # SOFT BLOCK: Warning level (requires confirmation)
                    elif safety_mgr.should_require_confirmation(metrics):
                        logger.warning("Transition requires user confirmation")

                        confirm_mgr = UserConfirmationManager(logger)
                        decision = confirm_mgr.request_confirmation(metrics, report)

                        logger.info(
                            f"User decision: {decision.action.value} ({decision.reason})"
                        )

                        if decision.action == TransitionAction.CANCEL:
                            trades_to_execute = {}
                            logger.info("All trades cancelled by user")
                        elif decision.action == TransitionAction.SELLS_ONLY:
                            trades_to_execute = safety_mgr.get_sells_only(
                                trades_to_execute
                            )
                            logger.info(
                                f"Proceeding with SELL orders only: {trades_to_execute}"
                            )
                        elif decision.action == TransitionAction.SCALE_DOWN:
                            trades_to_execute = safety_mgr.get_safe_trade_subset(
                                trades_to_execute, metrics
                            )
                            logger.info(f"Scaled down trades: {trades_to_execute}")
                        # else: PROCEED with original trades

                    else:
                        # SAFE or auto-proceed
                        if metrics.risk_level in [RiskLevel.SAFE, RiskLevel.INFO]:
                            logger.info(
                                "Transition safety check PASSED - proceeding with trades"
                            )
                        else:
                            logger.info(
                                f"Auto-proceeding (risk level: {metrics.risk_level.value})"
                            )

                    logger.info("=" * 70 + "\n")
                # === END TRANSITION SAFETY CHECK ===

                if trades_to_execute:
                    logger.info(
                        f"Trades to execute (including position cleanup): {trades_to_execute}"
                    )

                    # === ORDER ENTRY STRATEGY ===
                    # Initialize order scheduler for per-exchange timing
                    order_scheduler = OrderScheduler(logger)

                    # Generate submission schedule for all symbols
                    schedule = order_scheduler.get_submission_schedule(
                        list(trades_to_execute.keys()),
                        exchange_manager,
                        ORDER_SUBMISSION_TIMING,
                        ORDER_SUBMISSION_TIMING_BY_EXCHANGE,
                        dt.datetime.combine(today_date, dt.time.min),
                    )

                    # Log schedule summary
                    logger.info(order_scheduler.log_schedule_summary(schedule))

                    # Log order entry strategy configuration
                    logger.info("\n" + "=" * 60)
                    logger.info("ORDER ENTRY STRATEGY")
                    logger.info("=" * 60)
                    logger.info(f"  Order Type: {ORDER_TYPE}")
                    logger.info(f"  Limit Price Offset: {LIMIT_PRICE_OFFSET}")
                    logger.info(f"  All-or-None (AON): {ORDER_ALL_OR_NONE}")
                    logger.info(f"  Default Timing: {ORDER_SUBMISSION_TIMING}")
                    if ORDER_SUBMISSION_TIMING_BY_EXCHANGE:
                        logger.info(
                            f"  Exchange Overrides: {ORDER_SUBMISSION_TIMING_BY_EXCHANGE}"
                        )
                    logger.info("=" * 60 + "\n")

                    if ORDER_TYPE == "LIMIT":
                        pending_order_book.replace_with_trades(
                            trades_to_execute, exchange_manager
                        )
                        grouped_pending = pending_order_book.list_by_exchange()

                        now_utc = dt.datetime.now(pytz.UTC)
                        reference_datetime = dt.datetime.combine(
                            today_date, dt.time.min
                        )
                        exchanges_by_time = []

                        for exchange in grouped_pending:
                            wait_minutes = LIMIT_WAIT_AFTER_OPEN_BY_EXCHANGE.get(
                                exchange, LIMIT_WAIT_AFTER_OPEN_MINUTES
                            )
                            submission_time = (
                                order_scheduler.calculate_open_plus_minutes(
                                    exchange,
                                    wait_minutes,
                                    reference_datetime,
                                )
                            )
                            if submission_time is None:
                                submission_time = now_utc
                            exchanges_by_time.append(
                                (submission_time, exchange, wait_minutes)
                            )

                        exchanges_by_time.sort(key=lambda x: x[0])

                        submitted_count = 0
                        failed_count = 0

                        for (
                            submission_time,
                            exchange,
                            wait_minutes,
                        ) in exchanges_by_time:
                            wait_seconds = int(
                                (
                                    submission_time - dt.datetime.now(pytz.UTC)
                                ).total_seconds()
                            )
                            if wait_seconds > 0:
                                logger.info(
                                    f"{exchange}: waiting {wait_seconds}s for open+{wait_minutes}m LIMIT execution window"
                                )
                                time.sleep(wait_seconds)

                            exchange_orders = sorted(
                                grouped_pending.get(exchange, []),
                                key=lambda item: item[1].get("quantity", 0),
                            )

                            for symbol, pending_info in exchange_orders:
                                quantity_to_trade = pending_info.get("quantity", 0)
                                if (
                                    symbol in BLACKLISTED_SYMBOLS
                                    and quantity_to_trade > 0
                                ):
                                    logger.warning(
                                        f"Attempted to BUY blacklisted symbol {symbol} - skipping"
                                    )
                                    pending_order_book.mark_failed(
                                        symbol, "Blacklisted symbol"
                                    )
                                    failed_count += 1
                                    continue

                                contract = subscribed_contracts.get(symbol)
                                if not contract:
                                    if symbol not in active_symbols:
                                        logger.info(
                                            f"Creating contract for orphaned position {symbol}"
                                        )
                                        contract = create_contract(symbol)
                                        subscribed_contracts[symbol] = contract
                                    else:
                                        pending_order_book.mark_failed(
                                            symbol, "Missing contract"
                                        )
                                        failed_count += 1
                                        continue

                                raw_quantity = abs(quantity_to_trade)
                                rounded_quantity = exchange_manager.round_to_lot_size(
                                    symbol, raw_quantity
                                )
                                if rounded_quantity == 0:
                                    pending_order_book.mark_failed(
                                        symbol, "Rounded quantity is zero"
                                    )
                                    failed_count += 1
                                    continue

                                order_action = (
                                    "BUY" if quantity_to_trade > 0 else "SELL"
                                )
                                ok, error_msg = limit_order_engine.submit_with_retry(
                                    symbol,
                                    contract,
                                    rounded_quantity,
                                    order_action,
                                    max_retries=MAX_LIMIT_RETRIES,
                                    fill_timeout_seconds=LIMIT_FILL_TIMEOUT_SECONDS,
                                    quote_timeout_seconds=LIMIT_QUOTE_TIMEOUT_SECONDS,
                                    price_strategy=LIMIT_PRICE_STRATEGY,
                                    abs_offset=LIMIT_PRICE_OFFSET,
                                    pct_offset=LIMIT_PRICE_OFFSET_PCT,
                                )

                                if ok:
                                    pending_order_book.mark_submitted(symbol)
                                    submitted_count += 1
                                else:
                                    pending_order_book.mark_failed(
                                        symbol, error_msg or "Unknown error"
                                    )
                                    failed_count += 1

                        logger.info(
                            f"LIMIT execution complete: {submitted_count} submitted, {failed_count} failed"
                        )
                        if failed_count == 0:
                            pending_order_book.clear()
                        else:
                            pending_order_book.persist()
                    else:
                        # Sort trades: sells first (to free up capital), then buys
                        sorted_trades = sorted(
                            trades_to_execute.items(), key=lambda x: x[1]
                        )

                        for symbol, quantity_to_trade in sorted_trades:
                            if symbol in BLACKLISTED_SYMBOLS and quantity_to_trade > 0:
                                logger.warning(
                                    f"Attempted to BUY blacklisted symbol {symbol} - skipping"
                                )
                                continue

                            should_submit, timing_reason = (
                                order_scheduler.should_submit_now(symbol, schedule)
                            )
                            if not should_submit:
                                wait_seconds = order_scheduler.get_wait_time_seconds(
                                    symbol, schedule
                                )
                                if wait_seconds > 0 and wait_seconds < 3600:
                                    logger.info(f"{symbol}: {timing_reason}")
                                    logger.info(
                                        f"  Waiting {wait_seconds}s for scheduled submission..."
                                    )
                                    time.sleep(wait_seconds)
                                elif wait_seconds >= 3600:
                                    logger.warning(
                                        f"{symbol}: Scheduled time too far ({wait_seconds}s). Proceeding immediately."
                                    )

                            contract = subscribed_contracts.get(symbol)
                            if not contract:
                                if symbol not in active_symbols:
                                    logger.info(
                                        f"Creating contract for orphaned position {symbol}"
                                    )
                                    contract = create_contract(symbol)
                                    subscribed_contracts[symbol] = contract
                                else:
                                    logger.error(
                                        f"Contract not found for symbol {symbol}. Cannot place trade."
                                    )
                                    continue

                            raw_quantity = abs(quantity_to_trade)
                            rounded_quantity = exchange_manager.round_to_lot_size(
                                symbol, raw_quantity
                            )
                            if rounded_quantity == 0:
                                logger.warning(
                                    f"Quantity for {symbol} rounded to 0 (original: {raw_quantity}). Skipping trade."
                                )
                                continue

                            order_action = "BUY" if quantity_to_trade > 0 else "SELL"
                            order = Order()
                            order.action = order_action
                            order.totalQuantity = rounded_quantity
                            order.tif = "DAY"
                            order.transmit = True

                            if ORDER_TYPE == "MIDPRICE":
                                order.orderType = "MIDPRICE"
                            else:
                                order.orderType = "MKT"

                            order_id = ib_client.allocate_order_id()
                            # Determine order class for journal tagging
                            is_orphan = symbol not in active_symbols
                            order_class = "orphan_cleanup" if is_orphan else "stock"
                            order_guard.submit_stock_order(
                                symbol,
                                contract,
                                order,
                                order_id,
                                order_class=order_class,
                            )
                            time.sleep(0.5)

                        logger.info(
                            f"\nAll {len(trades_to_execute)} orders submitted. Waiting 3s for broker acknowledgment..."
                        )
                        time.sleep(3)
                else:
                    logger.info(
                        "No signal-based trades needed (all positions match signals)."
                    )

                # --- Wait for Stock Trades to Fill Before Carry Trade ---
                if trades_to_execute:
                    # Smart monitoring: polls every 10s, max 5 minutes
                    all_filled = wait_for_orders_to_fill(
                        order_manager, logger, timeout=300, poll_interval=10
                    )

                    if not all_filled:
                        logger.warning(
                            "Some orders didn't fill within 5 minutes. Proceeding with carry trade anyway."
                        )

                    # Revision Protocol: record post-rebalance equity event.
                    record_equity_event(
                        portfolio_manager,
                        region=region,
                        event="post_rebalance",
                        logger=logger,
                    )
                    evaluate_and_apply(
                        portfolio_manager,
                        region=region,
                        logger=logger,
                    )
                    if is_hard_kill_active():
                        logger.error(
                            "HARD KILL fired post-rebalance. Aborting carry trade phase."
                        )
                        return

                    # Refresh currency balances AFTER stock trades
                    # Use ALL expected currencies (stock + carry trade config) to detect debts
                    logger.info("\nRefreshing currency balances after stock trades...")
                    refresh_currencies = set(
                        forex_manager.get_expected_currencies_from_config()
                    )
                    if CASH_REBALANCING_MODE in ("phase1", "phase2"):
                        refresh_currencies.update(
                            CASH_PORTFOLIO_CONFIG.get("exotic_routing", {}).keys()
                        )
                        refresh_currencies.add(
                            CASH_PORTFOLIO_CONFIG.get("funding_currency", "USD")
                        )
                        refresh_currencies.add(
                            CASH_PORTFOLIO_CONFIG.get("carry_currency", "JPY")
                        )
                    ib_client.request_currency_balances(
                        currencies=sorted(refresh_currencies)
                    )

                # --- Carry Trade Strategy (after stock trades complete) ---
                # Dispatch based on CASH_REBALANCING_MODE feature flag
                logger.info("\n" + "=" * 60)
                logger.info(f"EXECUTING CARRY TRADE (mode: {CASH_REBALANCING_MODE})")
                logger.info("=" * 60)

                try:
                    if (
                        CASH_REBALANCING_MODE in ("phase1", "phase2")
                        and cash_portfolio_manager is not None
                    ):
                        # New cash portfolio engine
                        if CASH_REBALANCING_MODE == "phase1":
                            # Phase 1 only: exotic cleanup + static JPY conversion
                            carry_results = (
                                cash_portfolio_manager.run_phase1_exotic_cleanup(
                                    portfolio_manager.account_values
                                )
                            )
                            logger.info(
                                f"Phase 1 complete: {len(carry_results.get('executed', []))} orders"
                            )
                        else:
                            # Phase 2: full pipeline (exotic cleanup + ML carry trade)
                            carry_results = cash_portfolio_manager.run_full_rebalancing(
                                portfolio_manager.account_values
                            )
                            p1 = carry_results.get("phase1", {})
                            p2 = carry_results.get("phase2", {})
                            logger.info(
                                f"Phase 1: {len(p1.get('executed', []))} orders | "
                                f"Phase 2: {p2.get('converted', 0)} converted, "
                                f"{p2.get('reverted', 0)} reverted, "
                                f"{p2.get('held', 0)} held, "
                                f"{p2.get('forced', 0)} forced, "
                                f"{p2.get('failed', 0)} failed"
                            )
                    else:
                        # Legacy mode: original forex_manager
                        carry_trade_results = forex_manager.run_carry_trade_strategy(
                            portfolio_manager.account_values
                        )

                        if carry_trade_results.get("enabled"):
                            logger.info(
                                f"Carry trade executed: {len(carry_trade_results.get('executed', []))} orders"
                            )
                            if carry_trade_results.get("failed"):
                                logger.warning(
                                    f"Carry trade failures: {len(carry_trade_results['failed'])} orders"
                                )
                        else:
                            logger.info("JPY carry trade is disabled in config")

                except Exception as e:
                    logger.error(f"Error executing carry trade: {e}")
                    logger.error(traceback.format_exc())

                # Revision Protocol: record EOD equity event + final kill-switch
                # evaluation. This is the canonical "end of trading day" marker
                # that downstream attribution (Phase 1.3) keys off of.
                record_equity_event(
                    portfolio_manager,
                    region=region,
                    event="eod",
                    logger=logger,
                )
                evaluate_and_apply(
                    portfolio_manager,
                    region=region,
                    logger=logger,
                )

            # Connection assumed stable - no health checks needed

            # --- 11. Generate Periodic Report ---
            if time.time() - last_report_time >= report_interval:
                logger.info("Generating periodic trading report...")
                generate_report(
                    REPORT_FILE,
                    portfolio_manager.current_positions,
                    order_manager.get_open_orders(),
                    portfolio_manager.account_values,
                )

                last_report_time = time.time()

            time.sleep(daily_check_interval)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Unhandled error in main loop: {e}", exc_info=True)
    finally:
        logger.info("Closing any open orders before disconnecting...")
        for order_id in list(order_manager.get_open_orders().keys()):
            ib_client.cancel_order(order_id)
            time.sleep(0.5)

        logger.info(
            "Consider implementing a robust 'close all positions' routine here for production."
        )

        # Disconnect from IBKR
        ib_client.disconnect_ib()

        # Release the rotator-allocated client ID back to the shared pool so a
        # later session can reuse it. Best effort; never blocks shutdown.
        try:
            release_session_client_id(region_client_id)
        except Exception:
            pass

        # Close ZMQ handlers before closing socket to prevent errors
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except Exception:
                pass

        # Now safe to close ZMQ socket (may be None if the publisher could not
        # bind its port and we continued without external monitoring).
        try:
            if zmq_socket is not None:
                zmq_socket.close()
            if zmq_context is not None:
                zmq_context.term()
        except Exception:
            pass

        print("System shut down.")  # Use print since logger may be closed
        sys.exit(0)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IBKR Algorithmic Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Region-based Trading Sessions:
  Run separate sessions for different market regions at appropriate times.

  Example schedule for '1_HOUR_AFTER_OPEN':
    04:00 EST -> python main.py --region EUROPE  (EU markets open ~03:00-04:00 EST)
    10:30 EST -> python main.py --region US      (US market opens 09:30 EST)
    20:00 EST -> python main.py --region ASIA    (Tokyo opens 19:00 EST = 09:00 JST)
    07:00 EST -> python main.py --region CANADA  (Toronto opens 09:30 EST)
    19:00 EST -> python main.py --region OCEANIA (Sydney opens ~18:00 EST)

  Available regions: US, EUROPE, ASIA, MIDDLE_EAST, CANADA, OCEANIA, INDIA, ALL
        """,
    )

    parser.add_argument(
        "--region",
        type=str,
        default="ALL",
        choices=[
            "ALL",
            "US",
            "EUROPE",
            "ASIA",
            "MIDDLE_EAST",
            "CANADA",
            "OCEANIA",
            "INDIA",
        ],
        help="Trade only symbols from this region (default: ALL)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(region=args.region)
