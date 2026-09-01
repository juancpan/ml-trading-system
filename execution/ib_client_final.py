"""
Final production IB Client - clean, simple, no signal handlers.
Based on the working test_ibkr.py pattern.
"""

import os
import time
import threading
from datetime import datetime

# Disable proxy for localhost
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.utils import iswrapper

# Try to import OrderCancel for newer API versions (v10.19+)
# OrderCancel is in ibapi.order_cancel module
try:
    from ibapi.order_cancel import OrderCancel

    HAS_ORDER_CANCEL = True
except ImportError:
    HAS_ORDER_CANCEL = False
    OrderCancel = None

from config import IB_HOST, IB_PORT, IB_CLIENT_ID


class IBClient(EWrapper, EClient):
    """
    Production IB Client - clean and simple.
    """

    def __init__(self, portfolio_manager, order_manager, logger, client_id=None):
        EClient.__init__(self, self)
        self.portfolio_manager = portfolio_manager
        self.order_manager = order_manager
        self.logger = logger
        # No connection monitor - assuming stable connection

        # API client ID. Defaults to the legacy global IB_CLIENT_ID if a caller
        # does not supply one, but main.py passes a per-region ID so concurrent
        # region sessions never collide (IBKR error 326). See config.get_client_id.
        self.client_id = client_id if client_id is not None else IB_CLIENT_ID

        # Connection state
        self.nextValidOrderId = -1
        self.order_id_lock = threading.Lock()
        self.connected = False
        self.api_thread = None
        # Set by error() when the gateway reports the client ID is already in
        # use (326) or the socket closes mid-handshake (507 / -1). Used by
        # connect_and_run() to decide whether a connection failure is retryable.
        self.client_id_in_use = False

        # Position loading state (event-driven)
        self.positions_event = threading.Event()
        self.positions_loaded = False
        self.position_load_error = False  # Track errors during position loading

        # Currency balance tracking for multi-currency accounts (JPY carry trade)
        self.currency_balances = {}  # {currency: cash_balance} from reqAccountSummary
        self.account_summary_events = {}  # {reqId: Event()} for async tracking
        self.contract_details_mgr = None
        self.currency_converter = None
        self.market_data_mgr = None

        # Historical data request tracking (for IBKR EOD bar fallback)
        self._hist_data_bars = {}  # {reqId: [bar, ...]}
        self._hist_data_events = {}  # {reqId: Event()}
        self._hist_req_counter = 8000  # Start high to avoid collision with other reqIds

    def connect_and_run(self, max_retries=3, retry_backoff=7.0):
        """Connect to IB Gateway, retrying on a recoverable client-id collision.

        Error 326 ("client id is already in use") has two causes here:
          1. A legitimately-overlapping region session still holding this ID
             (mostly eliminated now that each region has a dedicated ID).
          2. A stale TCP session that has not yet torn down after the IB
             Gateway nightly restart (~23:45) — the gateway transiently reports
             326 until the old socket is reaped. This is retryable and is the
             documented behaviour (cf. IBKR docs "Broken API socket connection"
             and NautilusTrader PR #3796).

        We therefore retry up to ``max_retries`` times with a fixed backoff. If
        every attempt still hits 326 (or any other connect failure), we return
        False so the caller exits non-zero and the cron wrapper alerts — no
        silent skip, and no surprise fallback to a different client ID.
        """
        for attempt in range(1, max_retries + 1):
            if self._attempt_connect(attempt, max_retries):
                return True
            # Only the client-id-in-use case is worth retrying; a clean reset
            # of the flag lets us distinguish it on the next loop.
            if not self.client_id_in_use:
                # Non-326 failure (e.g. gateway down, port blocked): retrying
                # rarely helps and would only delay the alert. Fail fast.
                return False
            if attempt < max_retries:
                self.logger.warning(
                    "Client ID %s reported in use (326). Likely a stale session "
                    "after gateway restart or an overlapping run. Retry %d/%d in "
                    "%.0fs...",
                    self.client_id,
                    attempt,
                    max_retries,
                    retry_backoff,
                )
                self._reset_for_retry()
                time.sleep(retry_backoff)
            else:
                self.logger.critical(
                    "Client ID %s still in use after %d attempts. Giving up; "
                    "another session holds this ID. (IBKR error 326.)",
                    self.client_id,
                    max_retries,
                )
        return False

    def _reset_for_retry(self):
        """Tear down a half-open socket so the next connect attempt is clean."""
        self.client_id_in_use = False
        self.connected = False
        try:
            if self.isConnected():
                self.disconnect()
        except Exception:
            pass

    def _attempt_connect(self, attempt=1, max_retries=1):
        """
        Connect to IB Gateway with robust connection handling (single attempt).
        """
        self.logger.info(
            "Connecting to IB Gateway at %s:%s (Client ID: %s) [attempt %d/%d]",
            IB_HOST,
            IB_PORT,
            self.client_id,
            attempt,
            max_retries,
        )

        try:
            # Connect to IB Gateway
            self.connect(IB_HOST, IB_PORT, self.client_id)

            # Start the API thread for message processing IMMEDIATELY
            # This thread is required for the connection to work properly
            self.api_thread = threading.Thread(target=self.run, name="IB-API-Thread")
            self.api_thread.daemon = False
            self.api_thread.start()

            # Wait for connection to establish with proper state checking
            max_wait = 10  # Wait up to 10 seconds
            wait_increment = 0.1
            total_waited = 0

            self.logger.debug("Waiting for connection to establish...")
            while total_waited < max_wait:
                if self.isConnected():
                    # Give the connection a moment to fully stabilize
                    time.sleep(0.5)
                    break
                # Short-circuit: if the gateway already told us the client ID
                # is in use, there is no point waiting the full timeout — let
                # the retry loop handle it immediately.
                if self.client_id_in_use:
                    self.logger.error(
                        "Connect rejected: client ID %s already in use (326).",
                        self.client_id,
                    )
                    return False
                time.sleep(wait_increment)
                total_waited += wait_increment

            # Check if connected
            if not self.isConnected():
                if self.client_id_in_use:
                    self.logger.error(
                        "Connect rejected: client ID %s already in use (326).",
                        self.client_id,
                    )
                    return False
                self.logger.error(
                    "Failed to establish connection after %d seconds", max_wait
                )
                self.logger.error("Please verify:")
                self.logger.error("  1. IB Gateway is running")
                self.logger.error("  2. API connections are enabled in IB Gateway")
                self.logger.error("  3. Port %s is accessible", IB_PORT)
                self.logger.error("  4. No firewall is blocking the connection")
                return False

            # Mark as connected
            self.connected = True

            # Log connection info (now that we're connected)
            self.logger.info("Server Version: %s", self.serverVersion())
            self.logger.info("Connection Time: %s", self.twsConnectionTime())
            self.logger.info("Successfully connected to IB Gateway")

            # Request initial data with retry logic
            self.logger.debug("Requesting next valid order ID...")
            self.reqIds(-1)  # Request next valid order ID

            # Wait for nextValidId callback with timeout
            max_wait_order_id = 5
            wait_start = time.time()
            while (
                self.nextValidOrderId == -1
                and (time.time() - wait_start) < max_wait_order_id
            ):
                time.sleep(0.1)

            if self.nextValidOrderId == -1:
                self.logger.warning("Did not receive valid order ID within timeout")

            # Request account updates only if still connected
            if self.isConnected():
                self.logger.debug("Requesting account updates...")
                self.request_account_updates()
                time.sleep(2)  # Give time for initial account data
            else:
                self.logger.error("Lost connection before requesting account updates")
                return False

            self.logger.info(
                "Connection setup complete. Next Order ID: %s",
                self.nextValidOrderId if self.nextValidOrderId != -1 else "pending",
            )
            return True

        except Exception as e:
            self.logger.error("Connection error: %s", e)
            import traceback

            self.logger.error("Traceback: %s", traceback.format_exc())
            return False

    def request_account_updates(self):
        """Request account and position updates with connection verification."""
        if self.isConnected():
            self.logger.info("Requesting account updates")
            try:
                self.reqAccountUpdates(True, "")
            except Exception as e:
                self.logger.error("Error requesting account updates: %s", e)
        else:
            self.logger.warning("Cannot request account updates - not connected")

    def request_positions(self, timeout=30, max_retries=3):
        """
        PRODUCTION-GRADE position loading with retry logic and error recovery.
        CRITICAL: Must be called at startup before trading to prevent duplicate orders!

        Implements:
        - Retry logic with exponential backoff (3 attempts)
        - cancelPositions() before reqPositions() to clear stale subscriptions
        - Error state tracking to distinguish timeout from errors
        - Cleanup after successful load

        Args:
            timeout: Maximum seconds to wait per attempt (default: 30)
            max_retries: Number of attempts before giving up (default: 3)

        Returns:
            bool: True if positions loaded successfully, False after all retries exhausted
        """
        for attempt in range(max_retries):
            if attempt > 0:
                backoff = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s, 4s
                self.logger.warning(
                    f"Position loading retry {attempt + 1}/{max_retries} (backoff: {backoff}s)"
                )
                time.sleep(backoff)

            if self._request_positions_single_attempt(timeout):
                self.logger.info(
                    f"✓ Position loading succeeded on attempt {attempt + 1}"
                )
                return True

            self.logger.warning(
                f"Position loading attempt {attempt + 1}/{max_retries} failed"
            )

        # All retries exhausted
        self.logger.critical(
            f"FATAL: Position loading FAILED after {max_retries} attempts"
        )
        self.logger.critical(
            "Cannot proceed with trading - would risk duplicate orders!"
        )
        return False

    def _request_positions_single_attempt(self, timeout=30):
        """
        Single attempt to load positions.
        Handles subscription cleanup, error states, and timeout.

        Returns:
            bool: True if successful, False on timeout or error
        """
        if not self.isConnected():
            self.logger.error("Cannot request positions - not connected")
            return False

        # CRITICAL FIX #1: Cancel any existing subscription first
        # Without this, positionEnd() won't fire on repeated calls
        try:
            self.logger.debug("Canceling any existing position subscription...")
            self.cancelPositions()
            time.sleep(0.2)  # Allow cleanup to complete
        except Exception as e:
            self.logger.debug(f"No existing subscription to cancel: {e}")

        # Reset state for fresh attempt
        self.positions_event.clear()
        self.positions_loaded = False
        self.position_load_error = False

        # CRITICAL: Enable position=0 filtering mode in portfolio_manager
        # This prevents both position() AND updatePortfolio() callbacks with position=0
        # from clearing the dictionary during initial load
        self.portfolio_manager.initial_position_load_mode = True

        # Subscribe to positions
        try:
            self.logger.info("Calling reqPositions()...")
            self.reqPositions()
        except Exception as e:
            self.logger.error(f"Exception calling reqPositions(): {e}")
            self.portfolio_manager.initial_position_load_mode = False  # Reset on error
            return False

        # Wait for positionEnd() callback or error with timeout
        self.logger.info(f"Waiting for positions (timeout: {timeout}s)...")
        if self.positions_event.wait(timeout=timeout):
            # Event was set - check why
            if self.position_load_error:
                self.logger.error("Position loading failed due to error callback")
                self.portfolio_manager.initial_position_load_mode = False  # Reset flag
                return False

            if self.positions_loaded:
                # CRITICAL: Disable position=0 filtering mode AFTER loading complete
                self.portfolio_manager.initial_position_load_mode = False

                # Reconcile any stray wrong-exchange keys (e.g. XYZ.PA -> XYZ.MI)
                # left by the historical EUR->SBF inference bug. Dry-run report
                # first (visible in the log), then apply. See MEMORY.md
                # "European position stored under wrong exchange key (.PA)".
                try:
                    planned = self.portfolio_manager.reconcile_position_keys(dry_run=True)
                    if planned:
                        self.logger.warning(
                            "Position-key reconciler will fix %d stray key(s): %s",
                            len(planned),
                            ", ".join(f"{c['stray_key']}->{c['canonical_key']}" for c in planned),
                        )
                        self.portfolio_manager.reconcile_position_keys(dry_run=False)
                except Exception as exc:
                    self.logger.error("Position-key reconciler failed (non-fatal): %s", exc)

                position_count = len(self.portfolio_manager.current_positions)
                self.logger.info(f"✓ All positions loaded: {position_count} positions")

                # Log each position for verification
                for (
                    symbol,
                    pos_data,
                ) in self.portfolio_manager.current_positions.items():
                    pos = float(pos_data.get("position", 0))
                    if pos != 0:
                        self.logger.info(f"  {symbol}: {pos} shares")

                return True
            else:
                self.logger.error(
                    "Event set but positions_loaded flag not True (unknown state)"
                )
                self.portfolio_manager.initial_position_load_mode = False  # Reset flag
                return False
        else:
            # Timeout occurred
            self.logger.error(
                f"TIMEOUT: Positions not received within {timeout} seconds"
            )
            self.portfolio_manager.initial_position_load_mode = False  # Reset flag
            return False

    def request_currency_balances(self, currencies=None, timeout=10):
        """
        Request per-currency cash balances using reqAccountSummary.
        CRITICAL: This is the ONLY way to detect GBP, JPY, EUR cash debts.
        reqAccountUpdates() only returns USD - cannot detect -£31,732 GBP debt!

        Official IBKR API Method:
        reqAccountSummary(reqId, group, tags)

        Args:
            currencies: List of currencies to check (default: auto-detect from positions)
            timeout: Timeout per currency request

        Returns:
            Dict of {currency: cash_balance}
        """
        if not self.isConnected():
            self.logger.error("Cannot request currency balances - not connected")
            return {}

        # Auto-detect currencies from current positions if not provided
        if currencies is None:
            currencies = set()
            for symbol, pos_data in self.portfolio_manager.current_positions.items():
                contract = pos_data.get("contract")
                if contract:
                    curr = getattr(contract, "currency", "USD")
                    currencies.add(curr)
            currencies.add("USD")  # Always include USD
            currencies = list(currencies)

        self.logger.info(f"Requesting currency balances for: {currencies}")
        self.currency_balances = {}

        # CRITICAL: Request ONE currency at a time, cancel before next
        # IBKR allows max 2 concurrent reqAccountSummary subscriptions
        # Sequential request-cancel-request pattern stays under limit
        reqId = 9001  # Use same ID for all (cancel between)

        for currency in currencies:
            self.logger.info(f"  Requesting {currency} cash balance...")
            event = threading.Event()
            self.account_summary_events[reqId] = event

            try:
                # Request this currency's balance
                self.reqAccountSummary(reqId, "All", f"$LEDGER:{currency}")

                # Wait for accountSummaryEnd callback
                if not event.wait(timeout=timeout):
                    self.logger.warning(f"  Timeout waiting for {currency} balance")

                # CRITICAL: Cancel THIS request BEFORE starting next
                # Prevents "Maximum number of requests exceeded" error
                self.cancelAccountSummary(reqId)
                time.sleep(0.2)  # Allow cancel to complete

            except Exception as e:
                self.logger.error(f"  Error requesting {currency}: {e}")
                # Try to cancel even on error
                try:
                    self.cancelAccountSummary(reqId)
                    time.sleep(0.1)
                except:
                    pass

        self.logger.info(f"✓ Currency balances loaded: {self.currency_balances}")
        return self.currency_balances

    def disconnect_ib(self):
        """Disconnect from IB Gateway with proper cleanup."""
        try:
            if self.isConnected():
                self.logger.info("Disconnecting from IB Gateway")
                # Stop account updates
                try:
                    self.reqAccountUpdates(False, "")
                    time.sleep(0.5)
                except Exception as e:
                    self.logger.warning("Error stopping account updates: %s", e)

                # Disconnect from IB
                try:
                    self.disconnect()
                    self.connected = False
                except Exception as e:
                    self.logger.warning("Error during disconnect call: %s", e)

                # Wait for connection to close
                max_wait = 3
                wait_start = time.time()
                while self.isConnected() and (time.time() - wait_start) < max_wait:
                    time.sleep(0.1)

            self.logger.info("Disconnected from IB Gateway")

            # Wait for API thread to finish
            if self.api_thread and self.api_thread.is_alive():
                self.logger.debug("Waiting for API thread to terminate...")
                self.api_thread.join(timeout=2)
                if self.api_thread.is_alive():
                    self.logger.warning("API thread did not terminate gracefully")

        except Exception as e:
            self.logger.error("Error during disconnect: %s", e)

    # === Callback Methods ===

    @iswrapper
    def error(
        self,
        reqId: int,
        errorTime: int,
        errorCode: int,
        errorString: str,
        advancedOrderRejectJson="",
    ):
        """Handle errors from the API - Updated for IBAPI 10.37 with errorTime parameter."""
        # Farm connection OK messages (not really errors)
        if errorCode in [2104, 2106, 2158]:
            self.logger.info("Farm connection OK (%d): %s", errorCode, errorString)
        # Client ID collision / broken socket during handshake. 326 is the
        # gateway explicitly rejecting a duplicate client ID; 507 (Java "Bad
        # Message") is the socket-EOF variant (per IBKR docs). Previously 326
        # fell through to the benign "System message" branch (errorCode < 1000)
        # and was never flagged — which is exactly how US/CANADA silently
        # failed to trade. Mark it so connect_and_run() can retry/fail loudly.
        # (errorCode -1 is deliberately NOT included: in the Python API it also
        # accompanies benign farm-status notifications and would false-positive.)
        elif errorCode in [326, 507]:
            self.client_id_in_use = True
            self.logger.error(
                "Connection error (%d): %s — client ID may be in use or socket "
                "closed during handshake.",
                errorCode,
                errorString,
            )
        # Real connection errors
        elif errorCode == 502:
            self.logger.critical("Cannot connect to TWS: %s", errorString)
        elif errorCode == 504:
            self.logger.error("Not connected to TWS/Gateway")
        # Farm connection issues
        elif errorCode in [2103, 2105, 2107, 2157]:
            self.logger.warning(
                "Farm connection issue (%d): %s", errorCode, errorString
            )
        # System messages
        elif errorCode < 1000:
            self.logger.info("System message (%d): %s", errorCode, errorString)
        # Other warnings
        else:
            self.logger.warning("Warning (%d): %s", errorCode, errorString)

        # CRITICAL FIX: Unblock position loading on fatal errors
        # Without this, error during position load causes indefinite hang
        # Error codes: 502=Can't connect, 504=Not connected, 2109=Account access denied
        if errorCode in [502, 504, 2109] and not self.positions_loaded:
            self.logger.error(f"Fatal error during position loading - unblocking wait")
            self.position_load_error = True
            self.positions_event.set()  # Unblock with error state

        if hasattr(self, "market_data_mgr") and self.market_data_mgr:
            self.market_data_mgr.handle_error(reqId, errorCode, errorString)

        # Unblock historical data requests on terminal errors.
        # Without this, error 200 ("No security definition") causes a 30-second
        # timeout wait because historicalDataEnd is never called.
        # Error codes: 200=No security definition, 162=Historical data request
        # pacing violation, 321=Error validating request, 322=Error processing request
        if errorCode in (200, 162, 321, 322) and reqId in self._hist_data_events:
            self.logger.debug(
                f"Historical data request {reqId} failed (error {errorCode}). "
                f"Unblocking immediately."
            )
            self._hist_data_events[reqId].set()

    @iswrapper
    def nextValidId(self, orderId: int):
        """Receive the next valid order ID."""
        super().nextValidId(orderId)
        with self.order_id_lock:
            self.nextValidOrderId = max(self.nextValidOrderId, orderId)
        self.logger.info("Next valid order ID: %d", orderId)
        if self.order_manager and hasattr(self.order_manager, "set_next_order_id"):
            self.order_manager.set_next_order_id(orderId)
        elif self.order_manager:
            self.order_manager.next_order_id = orderId

    @iswrapper
    def updateAccountValue(self, key: str, value: str, currency: str, accountName: str):
        """Handle account value updates."""
        try:
            self.portfolio_manager.update_account_value(
                key, value, currency, accountName
            )
        except Exception as e:
            self.logger.error("Error updating account value %s: %s", key, e)

    @iswrapper
    def updatePortfolio(
        self,
        contract,
        position: float,
        marketPrice: float,
        marketValue: float,
        averageCost: float,
        unrealizedPNL: float,
        realizedPNL: float,
        accountName: str,
    ):
        """Handle position updates from reqAccountUpdates()."""
        try:
            # Pass all parameters to portfolio manager
            self.portfolio_manager.update_position(
                contract,
                position,
                marketPrice,
                marketValue,
                averageCost,
                unrealizedPNL,
                realizedPNL,
                accountName,
            )
            # Safe conversion for logging
            try:
                pos_num = float(position) if position not in ["", None] else 0.0
                avg_num = float(averageCost) if averageCost not in ["", None] else 0.0
            except (ValueError, TypeError):
                pos_num = 0.0
                avg_num = 0.0
            self.logger.debug(
                "Position update: %s qty=%.0f avg=$%.2f",
                contract.symbol,
                pos_num,
                avg_num,
            )
        except Exception as e:
            self.logger.error("Error updating position for %s: %s", contract.symbol, e)

    @iswrapper
    def position(self, account: str, contract, position: float, avgCost: float):
        """
        Handle position updates from reqPositions().
        More reliable for startup position loading than updatePortfolio().

        Official IBKR API Callback:
        position(string account, Contract contract, double position, double avgCost)
        """
        try:
            # CRITICAL FIX: Ignore position=0 during initial position loading
            # IBKR sends position=0 for previously held but now closed positions
            # During initial load, these are stale notifications that would clear the dict
            # During ongoing updates (from reqAccountUpdates), position=0 is valid (position closed)
            if position == 0:
                self.logger.debug(
                    f"Ignoring position=0 for {contract.symbol} during initial load (stale close notification)"
                )
                return  # DON'T process zero positions during initial load

            # Only process actual positions (position > 0)
            self.portfolio_manager.update_position(
                contract=contract,
                position=position,
                marketPrice=0.0,  # Not provided in position() callback
                marketValue=0.0,  # Not provided
                averageCost=avgCost,
                unrealizedPNL=0.0,  # Not provided
                realizedPNL=0.0,  # Not provided
                accountName=account,
            )
            self.logger.info(
                f"Position received: {contract.symbol} = {position} shares @ ${avgCost:.2f}"
            )
        except Exception as e:
            self.logger.error(f"Error processing position for {contract.symbol}: {e}")

    @iswrapper
    def positionEnd(self):
        """
        All positions received signal from reqPositions().
        This callback fires ONCE after initial position data transmission.

        Official IBKR API Callback:
        positionEnd()
        """
        super().positionEnd()
        self.positions_loaded = True
        self.positions_event.set()

        # Cancel subscription to prevent resource waste
        # After initial load, position updates come via updatePortfolio() from reqAccountUpdates()
        try:
            self.cancelPositions()
            self.logger.debug("Position subscription cancelled (cleanup)")
        except Exception as e:
            self.logger.debug(f"Could not cancel positions: {e}")

        position_count = len(self.portfolio_manager.current_positions)
        self.logger.info(f"All positions loaded. Total: {position_count} positions")

    @iswrapper
    def accountDownloadEnd(self, accountName: str):
        """Called when account download is complete from reqAccountUpdates()."""
        self.logger.info("Account download complete for: %s", accountName)

    @iswrapper
    def openOrder(self, orderId: int, contract, order, orderState):
        """Handle open order updates.

        Also registers pre-existing orders (from prior sessions or TWS manual)
        that are discovered via reqAllOpenOrders() at startup.
        """
        # Safe conversion for order quantity
        try:
            qty = (
                float(order.totalQuantity)
                if order.totalQuantity not in ["", None]
                else 0.0
            )
        except (ValueError, TypeError):
            qty = 0.0

        self.logger.info(
            "Open order %d: %s %.0f %s - %s",
            orderId,
            order.action,
            qty,
            contract.symbol,
            orderState.status,
        )
        if self.order_manager:
            # Register pre-existing orders not yet tracked by order_manager.
            # These come from reqAllOpenOrders() at startup (prior sessions)
            # or from TWS manual orders.
            if not self.order_manager.get_order_info(orderId):
                self.order_manager.track_order(orderId, contract, order)
                self.logger.info(
                    f"Registered pre-existing order {orderId} for {contract.symbol} "
                    f"({order.action} {qty:.0f})"
                )
            self.order_manager.update_order_status(
                orderId, orderState.status, 0, order.totalQuantity, 0.0
            )

    @iswrapper
    def orderStatus(
        self,
        orderId: int,
        status: str,
        filled: float,
        remaining: float,
        avgFillPrice: float,
        permId: int,
        parentId: int,
        lastFillPrice: float,
        clientId: int,
        whyHeld: str,
        mktCapPrice: float,
    ):
        """Handle order status updates."""
        # IBKR API sometimes passes numeric values as strings or special UNSET values
        # Safe conversion to handle all cases
        try:
            filled_num = float(filled) if filled not in ["", None] else 0.0
            remaining_num = float(remaining) if remaining not in ["", None] else 0.0
            total_num = filled_num + remaining_num
        except (ValueError, TypeError):
            filled_num = 0.0
            remaining_num = 0.0
            total_num = 0.0

        self.logger.info(
            "Order %d status updated to: %s. Filled: %.0f/%.0f",
            orderId,
            status,
            filled_num,
            total_num,
        )

        if self.order_manager:
            self.order_manager.update_order_status(
                orderId, status, filled, remaining, avgFillPrice, lastFillPrice
            )

    @iswrapper
    def contractDetails(self, reqId: int, contractDetails):
        """
        Handle contract details from IBKR API.

        Official IBKR API Callback:
        contractDetails(int reqId, ContractDetails contractDetails)

        Delegates to ContractDetailsManager if available.
        """
        if hasattr(self, "contract_details_mgr") and self.contract_details_mgr:
            self.contract_details_mgr.handle_contract_details(reqId, contractDetails)
        else:
            self.logger.debug(
                "Received contract details for reqId %d but no manager available", reqId
            )

    @iswrapper
    def contractDetailsEnd(self, reqId: int):
        """
        Handle contract details end from IBKR API.

        Official IBKR API Callback:
        contractDetailsEnd(int reqId)

        Delegates to ContractDetailsManager if available.
        """
        if hasattr(self, "contract_details_mgr") and self.contract_details_mgr:
            self.contract_details_mgr.handle_contract_details_end(reqId)
        else:
            self.logger.debug("Contract details end for reqId %d", reqId)

    @iswrapper
    def tickPrice(self, tickerId: int, tickType: int, price: float, attrib):
        """
        Handle tick price updates from IBKR API.

        Official IBKR API Callback:
        tickPrice(int tickerId, int tickType, double price, TickAttrib attribs)

        Delegates to CurrencyConverter for forex rate updates.
        """
        if hasattr(self, "currency_converter") and self.currency_converter:
            self.currency_converter.handle_tick_price(tickerId, tickType, price)

        if hasattr(self, "market_data_mgr") and self.market_data_mgr:
            self.market_data_mgr.handle_tick_price(tickerId, tickType, price)

    @iswrapper
    def tickSnapshotEnd(self, tickerId: int):
        """
        Handle tick snapshot end from IBKR API.

        Official IBKR API Callback:
        tickSnapshotEnd(int tickerId)

        Delegates to CurrencyConverter if available.
        """
        if hasattr(self, "currency_converter") and self.currency_converter:
            self.currency_converter.handle_tick_snapshot_end(tickerId)

        if hasattr(self, "market_data_mgr") and self.market_data_mgr:
            self.market_data_mgr.handle_tick_snapshot_end(tickerId)

    # ------------------------------------------------------------------
    # Historical data callbacks (IBKR EOD bar fallback for price fetching)
    # ------------------------------------------------------------------

    @iswrapper
    def historicalData(self, reqId: int, bar):
        """Receive a single historical bar from IBKR."""
        if reqId in self._hist_data_bars:
            self._hist_data_bars[reqId].append(bar)

    @iswrapper
    def historicalDataEnd(self, reqId: int, start: str, end: str):
        """Signal that all historical bars for this request have been received."""
        event = self._hist_data_events.get(reqId)
        if event:
            event.set()

    def request_historical_bars(
        self,
        contract,
        duration="2 D",
        bar_size="1 day",
        timeout=12.0,
        what_to_show=None,
    ):
        """Request historical OHLCV bars from IBKR.

        Returns a list of dicts with full OHLCV data for each bar, or None.
        Auto-detects whatToShow if not specified: 'TRADES' for stocks, 'MIDPOINT' for forex.

        Args:
            contract: IBKR Contract object.
            duration: Duration string (e.g., '2 D', '5 D').
            bar_size: Bar size (e.g., '1 day', '1 hour').
            timeout: Max seconds to wait for response.

        Returns:
            List of dicts [{'date', 'open', 'high', 'low', 'close', 'volume'}, ...]
            or None on failure.
        """
        if not self.connected:
            return None

        req_id = self._hist_req_counter
        self._hist_req_counter += 1

        self._hist_data_bars[req_id] = []
        self._hist_data_events[req_id] = threading.Event()

        # Auto-detect if not specified; forex CASH doesn't support TRADES
        if what_to_show is None:
            what_to_show = "MIDPOINT" if contract.secType == "CASH" else "TRADES"

        try:
            self.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime="",  # Empty = now
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                # useRTH: forex CASH trades ~24/5 on IDEALPRO, so use all hours
                # (matches the proven pattern in algos/common/ibkr_downloader.py).
                # Hardcoding 1 here caused intermittent empty-bar returns for FX
                # pairs when the daily cron fired outside "regular" hours — the
                # silent root cause of the previous IBKR-historical-fallback bug.
                useRTH=0 if contract.secType == "CASH" else 1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[],
            )
        except Exception as e:
            self.logger.warning(f"reqHistoricalData failed for {contract.symbol}: {e}")
            self._hist_data_bars.pop(req_id, None)
            self._hist_data_events.pop(req_id, None)
            return None

        # Wait for historicalDataEnd callback
        event = self._hist_data_events[req_id]
        if not event.wait(timeout=timeout):
            self.logger.warning(
                f"IBKR historical data timeout for {contract.symbol} (reqId={req_id})"
            )
            self._hist_data_bars.pop(req_id, None)
            self._hist_data_events.pop(req_id, None)
            return None

        raw_bars = self._hist_data_bars.pop(req_id, [])
        self._hist_data_events.pop(req_id, None)

        if not raw_bars:
            self.logger.warning(
                f"IBKR returned no historical bars for {contract.symbol}"
            )
            return None

        # Convert IBKR bar objects to plain dicts with full OHLCV
        bars = []
        for bar in raw_bars:
            bars.append(
                {
                    "date": str(bar.date),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume) if hasattr(bar, "volume") else 0,
                }
            )

        return bars

    def request_historical_bar(
        self, contract, duration="2 D", bar_size="1 day", timeout=12.0
    ):
        """Convenience: Request the most recent historical bar's close price.

        Thin wrapper over request_historical_bars() for backward compatibility
        with portfolio_manager price fetching.

        Returns:
            Dict with {'close': float, 'date': str} or None on failure.
        """
        bars = self.request_historical_bars(contract, duration, bar_size, timeout)
        if not bars:
            return None

        last_bar = bars[-1]
        if last_bar["close"] <= 0:
            self.logger.warning(
                f"IBKR historical bar for {contract.symbol} has invalid close: "
                f"{last_bar['close']}"
            )
            return None

        return {"close": last_bar["close"], "date": last_bar["date"]}

    def allocate_order_id(self) -> int:
        """Allocate the next local order ID in a thread-safe way."""
        with self.order_id_lock:
            if self.nextValidOrderId < 0:
                raise RuntimeError("No valid IBKR order ID available yet")
            order_id = self.nextValidOrderId
            self.nextValidOrderId += 1
        return order_id

    @iswrapper
    def accountSummary(
        self, reqId: int, account: str, tag: str, value: str, currency: str
    ):
        """
        Receive currency-specific account summary data.

        Official IBKR API Callback:
        accountSummary(int reqId, String account, String tag, String value, String currency)

        Used to get per-currency cash balances (GBP, JPY, EUR) for carry trade.
        """
        if tag == "CashBalance" and currency not in ["BASE"]:
            try:
                balance = float(value)
                self.currency_balances[currency] = balance
                self.logger.info(f"  Currency cash balance: {balance:.2f} {currency}")
            except (ValueError, TypeError) as e:
                self.logger.debug(f"Error parsing {tag} for {currency}: {e}")

    @iswrapper
    def accountSummaryEnd(self, reqId: int):
        """
        Account summary complete for this request.

        Official IBKR API Callback:
        accountSummaryEnd(int reqId)
        """
        if reqId in self.account_summary_events:
            self.account_summary_events[reqId].set()
        self.logger.debug(f"Account summary complete for reqId {reqId}")

    def request_all_open_orders(self, timeout: float = 5.0):
        """Request all open orders from IBKR backend.

        Discovers orders from prior sessions, manually placed TWS orders,
        and any other open orders for this account. Results arrive via
        openOrder() and orderStatus() callbacks, which feed into
        order_manager automatically.

        Should be called once at startup before any trading logic runs.

        Args:
            timeout: Seconds to wait for callbacks to arrive.
        """
        self.logger.info("Requesting all open orders from IBKR...")
        self.reqAllOpenOrders()
        time.sleep(timeout)
        if self.order_manager:
            n_open = len(self.order_manager.get_open_orders())
            self.logger.info(f"Open order discovery complete: {n_open} order(s) found")

    def place_order(self, contract: Contract, order: Order, order_id: int):
        """Place an order with IB."""
        if not self.connected:
            self.logger.error("Cannot place order - not connected")
            return False

        # Safe conversion for order quantity
        try:
            qty = (
                float(order.totalQuantity)
                if order.totalQuantity not in ["", None]
                else 0.0
            )
        except (ValueError, TypeError):
            qty = 0.0

        self.logger.info(
            "Placing order %d: %s %.0f %s", order_id, order.action, qty, contract.symbol
        )
        if self.order_manager:
            self.order_manager.track_order(order_id, contract, order)
        self.placeOrder(order_id, contract, order)
        return True

    def cancel_order(self, order_id: int):
        """
        Cancel an order (backward compatible with old and new IBKR API).

        Args:
            order_id: Order ID to cancel

        Note: Newer IBKR API (v10.19+) requires OrderCancel object for CME Rule 576 compliance.
              This method automatically detects API version and uses appropriate signature.
        """
        if self.connected:
            self.logger.info("Canceling order %d", order_id)

            try:
                # Verify connection state before attempting cancel
                if self.serverVersion() is None:
                    self.logger.warning(
                        f"Cannot cancel order {order_id} - serverVersion not available (connection may be disconnected)"
                    )
                    return

                # Check if OrderCancel is available (newer API)
                if HAS_ORDER_CANCEL:
                    # New API: requires OrderCancel object
                    order_cancel = OrderCancel()
                    # Optional fields for manual order compliance:
                    # order_cancel.manualOrderCancelTime = ""  # CME compliance
                    # order_cancel.extOperator = ""  # External operator
                    # order_cancel.manualOrderIndicator = 0  # Manual order indicator
                    self.cancelOrder(order_id, order_cancel)
                else:
                    # Old API: just pass order_id
                    # Try with empty string as fallback for intermediate versions
                    try:
                        self.cancelOrder(order_id, "")
                    except TypeError:
                        # Very old API: only order_id parameter
                        self.logger.warning(
                            "Using legacy cancelOrder API (order_id only)"
                        )
                        # Use reqGlobalCancel or direct cancellation
                        # Note: This might not work on very old API versions
                        self.logger.error(
                            "Order cancellation may not be supported on this API version"
                        )
            except TypeError as e:
                # Handle serverVersion() returning None or other type comparison errors
                self.logger.warning(
                    f"Cannot cancel order {order_id} due to connection state error: {e}"
                )
            except Exception as e:
                # Catch any other errors during cancellation (e.g., during shutdown)
                self.logger.warning(f"Error canceling order {order_id}: {e}")
        else:
            self.logger.error("Cannot cancel order - not connected")

    def request_next_valid_id(self):
        """Request the next valid order ID."""
        if self.connected:
            self.reqIds(-1)
            return True
        return False

    def keep_alive(self):
        """Send a keep-alive request."""
        if self.connected:
            self.reqCurrentTime()
