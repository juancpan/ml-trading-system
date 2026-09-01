"""
IBKR historical data downloader via native ibapi.

Downloads daily OHLCV data from Interactive Brokers TWS API using a dedicated
connection (clientId=10) separate from the production trading system (clientId=0).

Handles:
- Pacing compliance (conservative concurrent request dispatch)
- Contract creation for stocks via ExchangeManager (28+ exchanges)
- Forex contracts via IDEALPRO (MIDPOINT, free data)
- Multi-year downloads by chaining sequential "1 Y" requests
- Concurrent requests (up to 15 pending, configurable)
- Connection management with auto-reconnect
- Graceful degradation when IB Gateway is not available

Usage:
    downloader = IBKRDataDownloader(port=4001, client_id=10)
    if downloader.connect():
        df = downloader.download_single("SPY", "2020-01-01", "2025-01-01")
        batch = downloader.download_batch(["SPY", "NVDA", "AAPL"], "2024-01-01", "2025-01-01")
        downloader.disconnect()
"""

import json
import logging
import re
import socket
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

from algos.common.client_id_rotator import (
    allocate_client_id,
    release_client_id,
)

logger = logging.getLogger(__name__)

# Try to import ExchangeManager for stock contract creation
try:
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_PROJECT_ROOT / "execution") not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT / "execution"))
    from exchange_manager import ExchangeManager

    _HAS_EXCHANGE_MANAGER = True
except ImportError:
    _HAS_EXCHANGE_MANAGER = False
    logger.warning(
        "ExchangeManager not available; stock contracts will use basic SMART routing"
    )


# ===========================================================================
# CONFIGURATION
# ===========================================================================

IBKR_CONFIG = {
    "default_host": "127.0.0.1",
    "default_port": 4001,
    "default_client_id": 10,
    "connect_timeout": 20.0,  # Seconds to wait for connection handshake
    "request_timeout": 30.0,  # Seconds to wait per historical data request
    "dispatch_delay": 0.5,  # Seconds between dispatching new requests
    "max_concurrent": 15,  # Max pending requests (IBKR allows 50, we're conservative)
    "reconnect_delay": 5.0,  # Seconds before reconnection attempt
    "max_reconnect_attempts": 3,  # Max reconnection tries
}

# IBKR error codes that indicate pacing violations
_PACING_ERROR_CODES = {162}  # "Historical Market Data Service error"
_NO_DATA_MESSAGES = [
    "no data",
    "HMDS query returned no data",
    "No market data permissions",
    "No security definition",
]
_DISCONNECT_CODES = {1100, 1101, 1102, 2110}

# Forex currencies for pair detection
_FOREX_CURRENCIES = {
    "EUR",
    "USD",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "NZD",
    "CAD",
    "HKD",
    "SGD",
    "NOK",
    "SEK",
    "DKK",
    "PLN",
    "CZK",
    "HUF",
    "ILS",
    "MXN",
    "ZAR",
    "TRY",
    "INR",
    "RON",
    "AED",
    "SAR",
    "MYR",
    "BRL",
    "KRW",
    "TWD",
    "THB",
    "PHP",
}

_FOREX_SPLIT_PATTERN = re.compile(r"[\s/._-]+")


# ===========================================================================
# HELPERS
# ===========================================================================


def is_gateway_available(
    host: str = None, port: int = None, timeout: float = 1.0
) -> bool:
    """
    Non-blocking TCP socket check for IB Gateway availability.

    Args:
        host: Gateway host (default 127.0.0.1)
        port: Gateway port (default 4001)
        timeout: Socket timeout in seconds

    Returns:
        True if the port is reachable.
    """
    host = host or IBKR_CONFIG["default_host"]
    port = port or IBKR_CONFIG["default_port"]
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


def is_forex_ticker(ticker: str) -> bool:
    """
    Detect if a ticker represents a forex pair.

    Handles both yfinance format (EURUSD=X) and normalized format (EURUSD).

    Args:
        ticker: Ticker symbol

    Returns:
        True if the ticker is a forex pair.
    """
    return normalize_forex_ticker(ticker) is not None


def normalize_forex_ticker(ticker: str) -> Optional[str]:
    """Normalize forex symbols to yfinance-style BASEQUOTE=X.

    Accepts IBKR/common formats such as EURUSD, EURUSD=X, EUR/USD, EUR.USD.

    Args:
        ticker: Input ticker string.

    Returns:
        Normalized ticker as BASEQUOTE=X, or None if invalid.
    """
    if not ticker:
        return None

    text = ticker.strip().upper()
    if text.endswith("=X"):
        text = text[:-2]

    parts = [p for p in _FOREX_SPLIT_PATTERN.split(text) if p]
    if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
        base, quote = parts[0], parts[1]
    else:
        compact = "".join(parts)
        if len(compact) != 6 or not compact.isalpha():
            return None
        base, quote = compact[:3], compact[3:]

    if base in _FOREX_CURRENCIES and quote in _FOREX_CURRENCIES:
        return f"{base}{quote}=X"
    return None


def parse_forex_pair(ticker: str) -> tuple:
    """
    Parse a forex ticker into base and quote currencies.

    Args:
        ticker: Forex ticker (e.g., 'EURUSD=X', 'EURUSD', 'USDJPY=X')

    Returns:
        (base_currency, quote_currency) tuple. E.g., ('EUR', 'USD')
    """
    normalized = normalize_forex_ticker(ticker)
    if normalized is None:
        raise ValueError(f"Invalid forex ticker for IBKR: {ticker}")
    clean = normalized.replace("=X", "")
    return (clean[:3], clean[3:])


# ===========================================================================
# IBKR DATA DOWNLOADER
# ===========================================================================


class IBKRDataDownloader(EWrapper, EClient):
    """
    IBKR historical data client via native ibapi.

    Uses a dedicated connection (clientId=10 by default) to avoid
    conflicting with the production trading system (clientId=0).

    The download workflow is:
    1. Connect to IB Gateway
    2. Create a Contract for the ticker
    3. Call reqHistoricalData with appropriate parameters
    4. Collect bars via historicalData() callback
    5. Signal completion via historicalDataEnd() callback
    6. Return the collected data as a DataFrame
    """

    def __init__(self, host: str = None, port: int = None, client_id: int = None):
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.host = host or IBKR_CONFIG["default_host"]
        self.port = port or IBKR_CONFIG["default_port"]

        # Client id: an explicit value (caller override) is honoured as-is.
        # Otherwise allocate a unique, collision-checked id from the shared
        # rotator (registry + live probe) instead of the old static default
        # (clientId=10), which collided with other sessions and raised IBKR
        # error 326. Ids we allocate are released on disconnect_gateway().
        self._owns_client_id = False
        if client_id is not None:
            self.client_id = client_id
        else:
            self.client_id = allocate_client_id(
                host=self.host, port=self.port, label="ibkr_downloader"
            )
            self._owns_client_id = True

        # Request tracking
        self._next_req_id = 1000
        self._req_id_lock = threading.Lock()
        self._pending_data: dict = {}  # reqId -> list of bar dicts
        self._pending_events: dict = {}  # reqId -> threading.Event
        self._pending_errors: dict = {}  # reqId -> error string

        # Connection state
        self._connected = False
        self._connection_event = threading.Event()
        self._connect_error: Optional[str] = None
        self._api_thread: Optional[threading.Thread] = None

        # Exchange manager for stock contract creation
        self._exchange_manager = ExchangeManager() if _HAS_EXCHANGE_MANAGER else None

        # priceMagnifier cache (lazy-loaded from contract_details_cache.json)
        self._price_magnifiers: Optional[dict] = None

        # IBKR contract map (lazy-loaded from ibkr_contract_map.json)
        # Generated by scripts/resolve_ibkr_contracts.py via reqContractDetails.
        # Maps yfinance ticker -> {conId, symbol, secType, exchange, currency, priceMagnifier}
        # or null (ticker is unfetchable via IBKR).
        self._contract_map: Optional[dict] = None

    # =========================================================================
    # CONTRACT MAP (from resolve_ibkr_contracts.py)
    # =========================================================================

    def _load_contract_map(self) -> dict:
        """Load resolved IBKR contract map from ibkr_contract_map.json.

        Generated by scripts/resolve_ibkr_contracts.py via reqContractDetails.
        Each entry is the EXACT contract specification from IBKR, including conId.
        """
        if self._contract_map is not None:
            return self._contract_map

        self._contract_map = {}
        # ibkr_downloader.py is at algos/common/ibkr_downloader.py
        # Contract map is at data/market_data/ibkr_contract_map.json (project root)
        # .parent.parent.parent goes: common/ -> algos/ -> project_root/
        map_path = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "market_data"
            / "ibkr_contract_map.json"
        )
        if map_path.exists():
            try:
                with open(map_path) as f:
                    self._contract_map = json.load(f)
                n_resolved = sum(
                    1 for v in self._contract_map.values() if v is not None
                )
                n_null = sum(1 for v in self._contract_map.values() if v is None)
                logger.info(
                    "Loaded IBKR contract map: %d resolved, %d unfetchable",
                    n_resolved,
                    n_null,
                )
            except Exception as e:
                logger.warning("Could not load IBKR contract map: %s", e)
        else:
            logger.debug("No IBKR contract map found at %s", map_path)

        return self._contract_map

    # =========================================================================
    # PRICE MAGNIFIER
    # =========================================================================

    def _load_price_magnifiers(self) -> dict:
        """Load priceMagnifier map from contract_details_cache.json.

        The cache is maintained by execution/contract_details_manager.py
        and contains magnifier values from live IBKR contract details.
        """
        if self._price_magnifiers is not None:
            return self._price_magnifiers

        self._price_magnifiers = {}
        cache_path = (
            Path(__file__).resolve().parent.parent.parent
            / "execution"
            / "contract_details_cache.json"
        )
        if cache_path.exists():
            try:
                import json

                with open(cache_path) as f:
                    cache = json.load(f)
                for sym, details in cache.items():
                    mag = details.get("priceMagnifier", 1)
                    if mag != 1:
                        self._price_magnifiers[sym] = mag
                logger.debug(
                    "Loaded %d priceMagnifier entries from cache",
                    len(self._price_magnifiers),
                )
            except Exception as e:
                logger.warning("Could not load priceMagnifier cache: %s", e)

        return self._price_magnifiers

    def _get_price_magnifier(self, ticker: str) -> int:
        """Get priceMagnifier for a ticker.

        Resolution order:
        1. IBKR contract map (ibkr_contract_map.json) — from reqContractDetails
        2. contract_details_cache.json (from live IBKR sessions)
        3. Hardcoded fallback: all .TA (TASE) stocks are in Agorot (÷100)
        4. Default: 1 (no adjustment)
        """
        # 1. Contract map (most authoritative)
        contract_map = self._load_contract_map()
        if ticker in contract_map and contract_map[ticker] is not None:
            mag = contract_map[ticker].get("priceMagnifier", 1)
            if mag and mag > 1:
                return mag

        # 2. Contract details cache
        magnifiers = self._load_price_magnifiers()
        if ticker in magnifiers:
            return magnifiers[ticker]

        # 3. Hardcoded fallback
        if ticker.endswith(".TA"):
            return 100
        return 1

    # =========================================================================
    # CONNECTION
    # =========================================================================

    def connect_gateway(self) -> bool:
        """
        Connect to IB Gateway. Starts the API message processing thread.

        Returns:
            True if connected successfully, False otherwise.
        """
        if self._connected and self.isConnected():
            return True

        if not is_gateway_available(self.host, self.port):
            logger.warning(f"IB Gateway not reachable at {self.host}:{self.port}")
            return False

        # Retry with the same EClient instance can leave ibapi internals in a
        # bad state after handshake timeouts. Keep one attempt per instance;
        # callers should create a fresh downloader object for reconnect attempts.
        attempts = 1
        for attempt in range(1, attempts + 1):
            try:
                self._connection_event.clear()
                self._connect_error = None
                self.connect(self.host, self.port, self.client_id)

                # Start API message processing thread once per connect attempt
                self._api_thread = threading.Thread(target=self.run, daemon=True)
                self._api_thread.start()

                if not self._connection_event.wait(
                    timeout=IBKR_CONFIG["connect_timeout"]
                ):
                    connected_state = self.isConnected()
                    error_suffix = (
                        f", last_error={self._connect_error}"
                        if self._connect_error
                        else ""
                    )
                    logger.warning(
                        "Connection handshake timeout on attempt "
                        f"{attempt}/{attempts} (isConnected={connected_state}{error_suffix})"
                    )
                    try:
                        self.disconnect()
                    except Exception:
                        pass
                    self._connected = False
                    if attempt < attempts:
                        time.sleep(IBKR_CONFIG["reconnect_delay"])
                        continue
                    logger.error(
                        "Connection timeout: did not receive nextValidId after retries"
                    )
                    return False

                self._connected = True
                logger.info(
                    f"Connected to IB Gateway at {self.host}:{self.port} (clientId={self.client_id})"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Failed to connect to IB Gateway (attempt {attempt}/{attempts}): {e}"
                )
                try:
                    self.disconnect()
                except Exception:
                    pass
                self._connected = False
                if attempt < attempts:
                    time.sleep(IBKR_CONFIG["reconnect_delay"])

        return False

    def disconnect_gateway(self) -> None:
        """Clean disconnect from IB Gateway."""
        if self.isConnected():
            self.disconnect()
        self._connected = False
        logger.info("Disconnected from IB Gateway")
        # Release a rotator-allocated client id so the pool stays free. Only
        # release ids WE allocated (not caller-supplied overrides).
        if getattr(self, "_owns_client_id", False):
            try:
                release_client_id(self.client_id, host=self.host, port=self.port)
            except Exception as e:
                logger.warning("Failed to release client id %s: %s", self.client_id, e)
            finally:
                self._owns_client_id = False

    def _get_next_req_id(self) -> int:
        """Thread-safe request ID generation."""
        with self._req_id_lock:
            req_id = self._next_req_id
            self._next_req_id += 1
            return req_id

    # =========================================================================
    # ibapi CALLBACKS
    # =========================================================================

    def nextValidId(self, orderId: int) -> None:
        """Called when connection is established and ready for requests."""
        super().nextValidId(orderId)
        self._connection_event.set()
        logger.debug(f"nextValidId: {orderId}")

    def error(self, reqId: int, *args) -> None:
        """
        Handle IBKR errors and informational messages.

        ibapi 10.x+ changed the error callback signature to include errorTime:
            New: error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)
            Old: error(reqId, errorCode, errorString, advancedOrderRejectJson)
        We handle both by detecting argument types.
        """
        # Parse variable-length args to extract errorCode and errorString
        if len(args) >= 3 and isinstance(args[1], int):
            # New format: (errorTime, errorCode, errorString, ...)
            errorCode = args[1]
            errorString = str(args[2]) if len(args) > 2 else ""
        elif len(args) >= 2 and isinstance(args[0], int):
            # Old format: (errorCode, errorString, ...)
            errorCode = args[0]
            errorString = str(args[1]) if len(args) > 1 else ""
        else:
            # Unknown format, log and return
            logger.debug(f"Unknown error format: reqId={reqId}, args={args}")
            return

        # Informational messages (not real errors)
        if errorCode in {2104, 2106, 2158, 2119, 2174}:
            logger.debug(f"IBKR info [{errorCode}]: {errorString}")
            return

        # Capture connection-stage errors for handshake diagnosis
        if reqId < 0 and errorCode in {
            326,
            502,
            503,
            504,
            1300,
            1100,
            1101,
            1102,
            2110,
        }:
            self._connect_error = f"[{errorCode}] {errorString}"

        # Disconnect events
        if errorCode in _DISCONNECT_CODES:
            logger.warning(f"IBKR disconnect [{errorCode}]: {errorString}")
            self._connected = False
            # Signal all pending requests as failed
            for rid, event in list(self._pending_events.items()):
                self._pending_errors[rid] = f"Disconnected: {errorString}"
                event.set()
            return

        # Pacing violation or no-data for a specific request
        if reqId >= 0 and reqId in self._pending_events:
            error_lower = errorString.lower()
            is_no_data = any(msg in error_lower for msg in _NO_DATA_MESSAGES)
            is_pacing = errorCode in _PACING_ERROR_CODES and "pacing" in error_lower

            if is_no_data:
                logger.info(f"[reqId={reqId}] No data available: {errorString}")
                self._pending_errors[reqId] = f"NO_DATA: {errorString}"
                self._pending_events[reqId].set()
            elif is_pacing:
                logger.warning(f"[reqId={reqId}] Pacing violation: {errorString}")
                self._pending_errors[reqId] = f"PACING: {errorString}"
                self._pending_events[reqId].set()
            else:
                logger.warning(f"[reqId={reqId}] Error [{errorCode}]: {errorString}")
                self._pending_errors[reqId] = errorString
                self._pending_events[reqId].set()
        else:
            logger.debug(f"IBKR error [reqId={reqId}, code={errorCode}]: {errorString}")

    def historicalData(self, reqId: int, bar) -> None:
        """Collect each historical data bar."""
        if reqId in self._pending_data:
            self._pending_data[reqId].append(
                {
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": float(bar.volume) if bar.volume >= 0 else 0.0,
                }
            )

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        """Signal that all historical data bars have been received."""
        super().historicalDataEnd(reqId, start, end)
        if reqId in self._pending_events:
            self._pending_events[reqId].set()
            logger.debug(
                f"[reqId={reqId}] historicalDataEnd: {start} to {end}, bars={len(self._pending_data.get(reqId, []))}"
            )

    # =========================================================================
    # CONTRACT CREATION
    # =========================================================================

    def create_contract(self, ticker: str) -> Optional[Contract]:
        """
        Create an IBKR Contract for a ticker.

        Resolution order:
        1. IBKR contract map (ibkr_contract_map.json) — conId-based, definitive.
           Generated by scripts/resolve_ibkr_contracts.py via reqContractDetails.
           Returns None if ticker is marked as unfetchable (null in map).
        2. Forex pairs (EURUSD=X, USDJPY=X) -> CASH on IDEALPRO
        3. ExchangeManager suffix-based mapping (fallback for new tickers)
        4. Default: US stock on SMART

        Args:
            ticker: Ticker in yfinance format (e.g., 'SPY', 'EURUSD=X', '8058.T')

        Returns:
            ibapi.contract.Contract, or None if ticker is unfetchable.
        """
        # 1. Try resolved contract map (conId-based, definitive)
        contract_map = self._load_contract_map()
        if ticker in contract_map:
            info = contract_map[ticker]
            if info is None:
                # Ticker was resolved as unfetchable (no IBKR STK contract exists)
                logger.debug(
                    f"[{ticker}] Marked unfetchable in contract map — skipping"
                )
                return None
            contract = Contract()
            sec_type = info.get("secType", "STK")
            contract.symbol = info.get("symbol", "")
            contract.secType = sec_type
            contract.currency = info.get("currency", "USD")

            if sec_type == "CASH":
                # Forex: use IDEALPRO, no conId needed
                contract.exchange = "IDEALPRO"
            elif info.get("conId"):
                # Stock/ETF: use conId with the stored exchange.
                # Most exchanges work with SMART routing, but some (B3 Brazil,
                # and potentially others) require direct exchange routing.
                # Using the stored exchange from reqContractDetails is always
                # correct since IBKR told us exactly where this contract trades.
                contract.conId = info["conId"]
                stored_exchange = info.get("exchange", "SMART")
                # IBKR reqHistoricalData works best with the actual exchange
                # for conId-based requests. SMART routing fails for some
                # exchanges (B3, TADAWUL, etc.)
                contract.exchange = stored_exchange
            else:
                # Fallback: use the stored exchange
                contract.exchange = info.get("exchange", "SMART")

            logger.debug(
                f"[{ticker}] From contract map: secType={sec_type}, "
                f"symbol={info.get('symbol')}, exchange={contract.exchange}"
            )
            return contract

        # 2. Forex pairs
        if is_forex_ticker(ticker):
            base, quote = parse_forex_pair(ticker)
            contract = Contract()
            contract.symbol = base
            contract.secType = "CASH"
            contract.currency = quote
            contract.exchange = "IDEALPRO"
            logger.debug(f"Created forex contract: {base}/{quote} on IDEALPRO")
            return contract

        # 3. ExchangeManager (suffix-based, for tickers not yet in contract map)
        if self._exchange_manager is not None:
            contract = self._exchange_manager.create_contract(ticker)
            if contract.exchange not in ("SMART", "IDEALPRO", "FUNDSERV"):
                contract.primaryExchange = contract.exchange
                contract.exchange = "SMART"
            return contract

        # 4. Fallback: basic US stock contract
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        return contract

    # =========================================================================
    # SINGLE TICKER DOWNLOAD
    # =========================================================================

    def _request_historical_data(
        self,
        ticker: str,
        contract: Contract,
        end_dt: str,
        duration: str,
        bar_size: str = "1 day",
    ) -> Optional[pd.DataFrame]:
        """
        Make a single reqHistoricalData call and wait for completion.

        Args:
            ticker: Ticker symbol (for logging)
            contract: IBKR Contract object
            end_dt: End datetime string (YYYYMMDD HH:mm:ss or "")
            duration: Duration string (e.g., "1 Y", "6 M", "30 D")
            bar_size: Bar size (e.g., "1 day")

        Returns:
            DataFrame with [open, high, low, close, volume] or None on failure.
        """
        req_id = self._get_next_req_id()
        self._pending_data[req_id] = []
        self._pending_events[req_id] = threading.Event()
        self._pending_errors.pop(req_id, None)

        # Determine whatToShow based on security type
        what_to_show = "MIDPOINT" if contract.secType == "CASH" else "TRADES"
        use_rth = 0 if contract.secType == "CASH" else 1

        try:
            self.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[],
            )

            # Wait for completion
            if not self._pending_events[req_id].wait(
                timeout=IBKR_CONFIG["request_timeout"]
            ):
                logger.warning(f"[{ticker}] Request timeout (reqId={req_id})")
                self.cancelHistoricalData(req_id)
                return None

            # Check for errors
            if req_id in self._pending_errors:
                error_msg = self._pending_errors[req_id]
                if error_msg.startswith("NO_DATA"):
                    logger.info(f"[{ticker}] No historical data available from IBKR")
                elif error_msg.startswith("PACING"):
                    logger.warning(f"[{ticker}] IBKR pacing violation")
                    raise PacingViolation(error_msg)
                else:
                    logger.warning(f"[{ticker}] IBKR error: {error_msg}")
                return None

            # Convert bars to DataFrame
            bars = self._pending_data.get(req_id, [])
            if not bars:
                return None

            df = pd.DataFrame(bars)
            # Parse date column
            df["date"] = pd.to_datetime(df["date"], format="mixed")
            df = df.set_index("date")
            df.index.name = "date"
            # Strip timezone if present
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df

        finally:
            # Cleanup
            self._pending_data.pop(req_id, None)
            self._pending_events.pop(req_id, None)
            self._pending_errors.pop(req_id, None)

    def download_single(
        self, ticker: str, start: str, end: str, bar_size: str = "1 day"
    ) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a single ticker.

        For date ranges > 1 year, chains sequential "1 Y" requests backward
        from the end date.

        Args:
            ticker: Ticker symbol (yfinance format: 'SPY', 'EURUSD=X', '8058.T')
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            bar_size: Bar size (default "1 day")

        Returns:
            DataFrame with columns [open, high, low, close, volume] and DatetimeIndex,
            or None if download fails.
        """
        if not self._connected or not self.isConnected():
            if not self.connect_gateway():
                return None

        contract = self.create_contract(ticker)
        if contract is None:
            logger.info(f"[{ticker}] Unfetchable via IBKR (not in contract map)")
            return None

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        # Calculate how many 1Y chunks we need
        total_days = (end_dt - start_dt).days
        all_chunks = []

        if total_days <= 365:
            # Single request -- use YYYYMMDD-HH:MM:SS UTC format
            duration = f"{total_days} D" if total_days <= 365 else "1 Y"
            end_str = end_dt.strftime("%Y%m%d-%H:%M:%S")
            df = self._request_historical_data(
                ticker, contract, end_str, duration, bar_size
            )
            if df is not None and not df.empty:
                all_chunks.append(df)
        else:
            # Chain multiple 1Y requests backward from end date
            current_end = end_dt
            while current_end > start_dt:
                end_str = current_end.strftime("%Y%m%d-%H:%M:%S")
                remaining_days = (current_end - start_dt).days

                if remaining_days > 365:
                    duration = "1 Y"
                else:
                    duration = f"{remaining_days} D"

                logger.debug(f"[{ticker}] Requesting {duration} ending {end_str}")
                df = self._request_historical_data(
                    ticker, contract, end_str, duration, bar_size
                )

                if df is not None and not df.empty:
                    all_chunks.append(df)
                    # Move end date to the earliest date we got - 1 day
                    current_end = df.index.min() - timedelta(days=1)
                else:
                    break

                # Small delay between chained requests for the same ticker
                time.sleep(IBKR_CONFIG["dispatch_delay"])

        if not all_chunks:
            return None

        # Combine and deduplicate
        combined = pd.concat(all_chunks)
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        # Filter to requested date range
        combined = combined.loc[
            (combined.index >= start_dt) & (combined.index <= end_dt)
        ]

        # IBKR does NOT provide split/dividend-adjusted prices.
        # For forex, this is fine (no splits/dividends on currency pairs).
        # For stocks, adj_close = close is WRONG for historical periods with
        # splits/dividends. This is why stocks should use yfinance (which
        # provides proper Adj Close) as primary source, with IBKR only as
        # fallback. The update_market_data.py workflow enforces this routing.
        combined["adj_close"] = combined["close"]

        # Apply priceMagnifier correction.
        # IBKR returns prices in exchange-native units:
        #   .TA (TASE): Agorot (1/100 ILS)
        #   .L (LSE): some stocks in pence (1/100 GBP)
        # Divide OHLC + adj_close by magnifier to get standard currency units.
        magnifier = self._get_price_magnifier(ticker)
        if magnifier > 1:
            for col in ("open", "high", "low", "close", "adj_close"):
                if col in combined.columns:
                    combined[col] = combined[col] / magnifier
            logger.info(
                f"[{ticker}] Applied priceMagnifier /{magnifier} "
                f"(exchange-native → standard units)"
            )

        logger.info(
            f"[{ticker}] Downloaded {len(combined)} bars from IBKR ({start} to {end})"
        )
        return combined

    # =========================================================================
    # BATCH DOWNLOAD
    # =========================================================================

    def download_batch(
        self,
        tickers: list,
        start: str,
        end: str,
        bar_size: str = "1 day",
        max_concurrent: int = None,
    ) -> dict:
        """
        Download data for multiple tickers with controlled concurrency.

        Args:
            tickers: List of ticker symbols (yfinance format)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            bar_size: Bar size (default "1 day")
            max_concurrent: Override max concurrent requests

        Returns:
            Dict of {ticker: DataFrame}. Missing tickers are omitted.
        """
        if not self._connected or not self.isConnected():
            if not self.connect_gateway():
                return {}

        max_concurrent = max_concurrent or IBKR_CONFIG["max_concurrent"]
        results = {}
        semaphore = threading.Semaphore(max_concurrent)

        def _download_one(t: str) -> None:
            with semaphore:
                try:
                    df = self.download_single(t, start, end, bar_size)
                    if df is not None and not df.empty:
                        results[t] = df
                except PacingViolation:
                    logger.warning(f"[{t}] Pacing violation in batch, skipping")
                except Exception as e:
                    logger.error(f"[{t}] Error in batch download: {e}")
                finally:
                    time.sleep(IBKR_CONFIG["dispatch_delay"])

        threads = []
        for ticker in tickers:
            t = threading.Thread(target=_download_one, args=(ticker,))
            t.start()
            threads.append(t)
            time.sleep(IBKR_CONFIG["dispatch_delay"])

        for t in threads:
            t.join(timeout=IBKR_CONFIG["request_timeout"] * 10)

        logger.info(
            f"Batch download complete: {len(results)}/{len(tickers)} tickers succeeded"
        )
        return results


class PacingViolation(Exception):
    """Raised when IBKR returns a pacing violation error."""

    pass


class NoDataError(Exception):
    """Raised when IBKR has no historical data for the requested instrument."""

    pass
