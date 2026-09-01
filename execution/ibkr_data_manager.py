"""
IBKR Data Manager — Central historical data interface for IBKR.

Wraps IBKR's reqHistoricalData API into pandas DataFrames matching the schema
expected by DataManager, PortfolioManager, and the strategy layer:

    Index:   pd.DatetimeIndex (timezone-naive, daily dates)
    Columns: ["Open", "High", "Low", "Close", "Volume"]  (capitalized, float64)
    Sort:    Ascending by date
    Close:   Split+dividend adjusted (after priceMagnifier correction)

Replaces yfinance for all live-trading data needs.
"""

import time
import threading
import logging
from typing import Optional, Tuple, Dict

import pandas as pd
from ibapi.contract import Contract


class IBKRDataManager:
    """Fetches historical OHLCV data from IBKR as the primary data source.

    Replaces yfinance for all live trading data needs.
    """

    PACING_DELAY_SECONDS = 2.0  # IBKR limit: 60 requests per 10 minutes

    # Symbol mapping is centralized in exchange_manager.py (single source of truth).
    # No duplicate suffix table here — all resolution goes through exchange_manager.

    def __init__(
        self,
        ib_client,
        contract_details_mgr,
        exchange_manager,
        logger=None,
    ):
        """
        Args:
            ib_client:             IBClient instance (has ``request_historical_bars``).
            contract_details_mgr:  ContractDetailsManager (symbol → contract details cache).
            exchange_manager:      ExchangeManager (symbol → exchange/currency parsing).
            logger:                Optional logger; falls back to module-level logger.
        """
        self.ib = ib_client
        self.contract_details_mgr = contract_details_mgr
        self.exchange_manager = exchange_manager
        self.logger = logger or logging.getLogger(__name__)

        # Pacing state — guarded by _pacing_lock
        self._last_request_time: float = 0.0
        self._pacing_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical_bars(
        self,
        symbol: str,
        num_days: int = 200,
        bar_size: str = "1 day",
        timeout: float = 15.0,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV bars for *symbol* from IBKR.

        Returns a DataFrame with the standard schema or ``None`` on failure.

        Args:
            symbol:   yfinance-format ticker (e.g. ``"RST.TO"``, ``"ABCD.TA"``).
            num_days: Calendar-day look-back translated into an IBKR duration string.
            bar_size: IBKR bar-size setting (default ``"1 day"``).
            timeout:  Seconds to wait for IBKR response.

        Returns:
            pd.DataFrame | None
        """
        contract = self._build_contract(symbol)
        if contract is None:
            self.logger.error("Cannot build contract for %s — skipping", symbol)
            return None

        duration_str = self._days_to_duration(num_days)

        self._wait_for_pacing()

        bars = self.ib.request_historical_bars(
            contract,
            duration=duration_str,
            bar_size=bar_size,
            timeout=timeout,
        )

        # If STK fails and we don't have a conId, retry with primaryExchange set.
        # Many LSE ETPs/ETCs need this for disambiguation.
        if not bars and not contract.conId and contract.exchange != "SMART":
            self.logger.debug(
                "Retrying %s with primaryExchange=%s", symbol, contract.exchange
            )
            retry_contract = Contract()
            retry_contract.symbol = contract.symbol
            retry_contract.secType = "STK"
            retry_contract.exchange = "SMART"
            retry_contract.primaryExchange = contract.exchange
            retry_contract.currency = contract.currency

            self._wait_for_pacing()
            bars = self.ib.request_historical_bars(
                retry_contract,
                duration=duration_str,
                bar_size=bar_size,
                timeout=timeout,
            )

        if not bars:
            self.logger.warning("IBKR returned no bars for %s", symbol)
            return None

        df = self._bars_to_dataframe(bars)
        if df is None or df.empty:
            self.logger.warning("Failed to convert bars to DataFrame for %s", symbol)
            return None

        df = self._apply_price_magnifier(symbol, df)
        return df

    def fetch_latest_close(self, symbol: str) -> Optional[Tuple[float, str]]:
        """Fetch the most recent closing price for *symbol*.

        Returns:
            ``(close_price, date_string)`` or ``None`` on failure.
        """
        df = self.fetch_historical_bars(symbol, num_days=5, bar_size="1 day")
        if df is None or df.empty:
            return None

        last_close = float(df["Close"].iloc[-1])
        last_date = str(df.index[-1].date())
        return (last_close, last_date)

    def fetch_forex_rate(self, base: str, quote: str) -> Optional[float]:
        """Fetch the latest forex mid-rate from IBKR MIDPOINT data.

        Args:
            base:  Base currency (e.g. ``"USD"``).
            quote: Quote currency (e.g. ``"JPY"``).

        Returns:
            Mid-rate as a float, or ``None`` on failure.
        """
        contract = self._build_contract(symbol=None, sec_type="CASH")
        if contract is None:
            contract = Contract()
        # Override fields for forex pair
        contract.symbol = base
        contract.currency = quote
        contract.exchange = "IDEALPRO"
        contract.secType = "CASH"

        self._wait_for_pacing()

        bars = self.ib.request_historical_bars(
            contract,
            duration="2 D",
            bar_size="1 day",
            timeout=12.0,
        )

        if not bars:
            self.logger.warning("No forex bars for %s/%s", base, quote)
            return None

        last_bar = bars[-1]
        rate = float(last_bar["close"])
        if rate <= 0:
            self.logger.warning("Invalid forex rate %.6f for %s/%s", rate, base, quote)
            return None

        self.logger.info("Forex %s/%s = %.6f", base, quote, rate)
        return rate

    # ------------------------------------------------------------------
    # Contract building
    # ------------------------------------------------------------------

    def _build_contract(
        self, symbol: Optional[str], sec_type: str = "STK"
    ) -> Optional[Contract]:
        """Map a yfinance symbol to an IBKR ``Contract``.

        Resolution order:
        1. ``contract_details_mgr.get_details(symbol)`` — has conId, exchange, currency.
        2. ``exchange_manager.parse_symbol(symbol)`` — suffix-based lookup.
        3. ``SUFFIX_MAP`` fallback (this class).

        For forex (``sec_type="CASH"``), returns a bare Contract; the caller
        must fill ``symbol`` / ``currency`` / ``exchange``.
        """
        if sec_type == "CASH":
            contract = Contract()
            contract.secType = "CASH"
            return contract

        if symbol is None:
            self.logger.error("_build_contract called with symbol=None for STK")
            return None

        contract = Contract()
        contract.secType = "STK"

        # --- 1. Try contract_details_mgr (most authoritative) ---
        details = self.contract_details_mgr.get_details(symbol)
        if details:
            # Use conId when available — most reliable IBKR identifier
            if details.get("conId"):
                contract.conId = details["conId"]

            # Use exchange_manager for correct symbol string
            # (handles hyphens, share classes, overrides)
            try:
                ibkr_sym, _, _ = self.exchange_manager.parse_symbol(symbol)
            except Exception:
                ibkr_sym = symbol.split(".")[0]
            contract.symbol = ibkr_sym
            contract.exchange = details.get("exchange", "SMART")
            contract.currency = details.get("currency", "USD")
            self.logger.debug(
                "Built contract for %s from details cache (conId=%s, exchange=%s)",
                symbol,
                contract.conId,
                contract.exchange,
            )
            return contract

        # --- 2. Try exchange_manager.parse_symbol ---
        try:
            ibkr_symbol, exchange, currency = self.exchange_manager.parse_symbol(symbol)
            contract.symbol = ibkr_symbol
            contract.exchange = exchange
            contract.currency = currency
            self.logger.debug(
                "Built contract for %s from exchange_manager (%s on %s)",
                symbol,
                ibkr_symbol,
                exchange,
            )
            return contract
        except Exception as exc:
            self.logger.warning(
                "exchange_manager.parse_symbol failed for %s: %s", symbol, exc
            )

        # --- 3. Last resort: bare symbol on SMART ---
        # Both contract_details_mgr and exchange_manager failed.
        # Try with just the base symbol on SMART routing.
        contract.symbol = symbol.split(".")[0]
        contract.exchange = "SMART"
        contract.currency = "USD"
        self.logger.warning(
            "Built contract for %s using SMART fallback (no details, no exchange mapping). "
            "This may fail for non-US symbols.",
            symbol,
        )
        return contract

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def _bars_to_dataframe(self, bars: list) -> Optional[pd.DataFrame]:
        """Convert the list-of-dicts from ``request_historical_bars`` into
        a standardised DataFrame.

        Expected input format per bar::

            {'date': '2026-04-07', 'open': 12.5, 'high': 12.8,
             'low': 12.3, 'close': 12.6, 'volume': 1234}

        Returns:
            DataFrame with DatetimeIndex and columns
            ``["Open", "High", "Low", "Close", "Volume"]``, sorted ascending.
        """
        if not bars:
            return None

        df = pd.DataFrame(bars)

        # Rename lowercase IBKR keys to capitalised columns
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df.rename(columns=rename_map, inplace=True)

        # Parse date → DatetimeIndex (timezone-naive)
        df["Date"] = pd.to_datetime(df["date"])
        df.set_index("Date", inplace=True)
        df.index.name = None  # match yfinance convention
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Keep only the required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            self.logger.error("Missing columns after conversion: %s", missing)
            return None
        df = df[required]

        # Ensure correct dtypes
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        return df

    # ------------------------------------------------------------------
    # Price magnifier
    # ------------------------------------------------------------------

    def _apply_price_magnifier(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Divide OHLC columns by the priceMagnifier if > 1.

        IBKR returns prices in exchange-native units:
        - ``.TA`` stocks → Agorot (1/100 ILS), priceMagnifier = 100
        - ``.L``  stocks → pence  (1/100 GBP), priceMagnifier = 100
        - Most others    → 1 (no adjustment)

        Volume is NOT adjusted.
        """
        magnifier = self.contract_details_mgr.get_price_magnifier(symbol)
        if magnifier > 1:
            self.logger.debug("Applying priceMagnifier=%d to %s", magnifier, symbol)
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = df[col] / magnifier
        return df

    # ------------------------------------------------------------------
    # Pacing
    # ------------------------------------------------------------------

    def _wait_for_pacing(self) -> None:
        """Enforce a minimum gap of ``PACING_DELAY_SECONDS`` between
        successive IBKR historical-data requests.

        IBKR limit: 60 identical requests per 10 minutes → ~1 request / 10 s.
        A 2-second floor is conservative enough for non-identical requests
        while avoiding the "pacing violation" error.
        """
        with self._pacing_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self.PACING_DELAY_SECONDS:
                wait = self.PACING_DELAY_SECONDS - elapsed
                self.logger.debug("Pacing: sleeping %.2f s", wait)
                time.sleep(wait)
            self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _days_to_duration(num_days: int) -> str:
        """Convert a calendar-day count into an IBKR duration string.

        IBKR accepts:
        - ``"N D"`` for days  (max 365)
        - ``"N W"`` for weeks
        - ``"N M"`` for months
        - ``"N Y"`` for years (max 1)

        For simplicity, anything ≤ 365 days uses ``"N D"``;
        otherwise falls back to ``"1 Y"``.
        """
        if num_days <= 365:
            return f"{num_days} D"
        return "1 Y"
