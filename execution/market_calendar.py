"""
Market Calendar Manager for Multi-Exchange Holiday Detection

Uses exchange_calendars library to determine trading days per exchange.
Enables intelligent per-exchange trading - only skip symbols on closed exchanges,
not the entire portfolio when US markets are closed.

Supported Exchanges:
- SMART (XNYS) - NYSE (US)
- TSEJ (XTKS) - Tokyo Stock Exchange
- LSE (XLON) - London Stock Exchange
- SEHK (XHKG) - Hong Kong Stock Exchange
- NSE (XNSE) - India NSE
- BSE (XBOM) - India BSE
- ASX (XASX) - Australia
- SGX (XSES) - Singapore
- SBF (XPAR) - Paris (Euronext)
- IBIS/FWB2 (XFRA) - Germany (Frankfurt/Xetra)
- BM (XMAD) - Madrid
"""

from typing import Dict, List, Tuple, Optional
from datetime import date, timedelta, datetime
import logging

try:
    import exchange_calendars as xcals
    EXCHANGE_CALENDARS_AVAILABLE = True
except ImportError:
    EXCHANGE_CALENDARS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class MarketCalendarManager:
    """
    Manages exchange calendars for multi-exchange holiday detection.

    Provides per-exchange trading day checks instead of US-only.
    """

    # IBKR exchange code -> exchange_calendars ISO code
    EXCHANGE_TO_ISO = {
        'SMART': 'XNYS',    # NYSE (US) - default for SMART routing
        'TSEJ': 'XTKS',     # Tokyo Stock Exchange
        'LSE': 'XLON',      # London Stock Exchange
        'SEHK': 'XHKG',     # Hong Kong Stock Exchange
        'NSE': 'XBOM',      # India NSE (uses BSE calendar - same trading days)
        'BSE': 'XBOM',      # India BSE (Bombay)
        'ASX': 'XASX',      # Australia
        'SGX': 'XSES',      # Singapore
        'SBF': 'XPAR',      # Euronext Paris
        'IBIS': 'XFRA',     # Deutsche Börse XETRA
        'FWB2': 'XFRA',     # Frankfurt (maps to same as XETRA)
        'BM': 'XMAD',       # Bolsa de Madrid
        # Additional exchanges from config
        'BUX': 'XBUD',      # Budapest Stock Exchange (Hungary)
        'SFB': 'XSTO',      # Nasdaq Stockholm (Sweden)
        'TADAWUL': 'XSAU',  # Tadawul (Saudi Arabia)
        'PRA': 'XPRA',      # Prague Stock Exchange (Czech Republic)
        'TSE': 'XTSE',      # Toronto Stock Exchange (Canada)
        'BVME': 'XMIL',     # Borsa Italiana (Milan)
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize MarketCalendarManager.

        Args:
            logger: Logger instance for output
        """
        self.logger = logger or logging.getLogger(__name__)
        self._calendars: Dict[str, any] = {}  # Cache for loaded calendars
        self._manual_closures: Dict[str, List[str]] = {}

        if not EXCHANGE_CALENDARS_AVAILABLE:
            self.logger.warning(
                "exchange_calendars library not installed. "
                "Install with: pip install exchange_calendars>=4.12"
            )

        # Load manual closures from config if available
        self._load_manual_closures()

    def _load_manual_closures(self):
        """Load manual market closure overrides from config."""
        try:
            from config import MANUAL_MARKET_CLOSURES
            self._manual_closures = MANUAL_MARKET_CLOSURES
            if self._manual_closures:
                self.logger.info(f"Loaded manual market closures: {self._manual_closures}")
        except ImportError:
            self._manual_closures = {}

    def _get_calendar(self, iso_code: str):
        """
        Get or create a calendar for the given ISO code.

        Args:
            iso_code: Exchange calendar ISO code (e.g., 'XNYS', 'XTKS')

        Returns:
            exchange_calendars.ExchangeCalendar instance or None
        """
        if not EXCHANGE_CALENDARS_AVAILABLE:
            return None

        if iso_code not in self._calendars:
            try:
                self._calendars[iso_code] = xcals.get_calendar(iso_code)
            except Exception as e:
                self.logger.warning(f"Could not load calendar for {iso_code}: {e}")
                self._calendars[iso_code] = None

        return self._calendars[iso_code]

    def _get_iso_code(self, ibkr_exchange: str) -> str:
        """
        Convert IBKR exchange code to ISO code.

        Args:
            ibkr_exchange: IBKR exchange code (e.g., 'TSEJ', 'LSE')

        Returns:
            ISO code for exchange_calendars (e.g., 'XTKS', 'XLON')
        """
        return self.EXCHANGE_TO_ISO.get(ibkr_exchange, 'XNYS')

    def is_trading_day(self, exchange: str, check_date: date) -> bool:
        """
        Check if a specific exchange is open on a given date.

        Args:
            exchange: IBKR exchange code (e.g., 'TSEJ', 'LSE', 'SMART')
            check_date: Date to check

        Returns:
            True if the exchange is open, False if closed (holiday/weekend)
        """
        # Check manual closures first
        if exchange in self._manual_closures:
            date_str = check_date.strftime('%Y-%m-%d')
            if date_str in self._manual_closures[exchange]:
                self.logger.info(f"{exchange} manually marked as closed on {date_str}")
                return False

        # Weekend check (all exchanges closed)
        if check_date.weekday() in [5, 6]:
            return False

        # If exchange_calendars not available, assume open on weekdays
        if not EXCHANGE_CALENDARS_AVAILABLE:
            self.logger.debug(
                f"exchange_calendars not available, assuming {exchange} open on {check_date}"
            )
            return True

        iso_code = self._get_iso_code(exchange)
        calendar = self._get_calendar(iso_code)

        if calendar is None:
            # Unknown exchange - assume open on weekdays
            self.logger.debug(f"No calendar for {exchange}/{iso_code}, assuming open")
            return True

        try:
            is_session = calendar.is_session(check_date)
            return is_session
        except Exception as e:
            self.logger.warning(f"Error checking {exchange} on {check_date}: {e}")
            return True  # Assume open on error

    def filter_tradeable_symbols(
        self,
        symbols: List[str],
        check_date: date,
        exchange_manager
    ) -> Tuple[List[str], List[str]]:
        """
        Filter symbols into tradeable (open market) and skipped (closed market).

        Args:
            symbols: List of yfinance symbols (e.g., ['NVDA', '8002.T', 'III.L'])
            check_date: Date to check
            exchange_manager: ExchangeManager instance for symbol->exchange mapping

        Returns:
            Tuple of (tradeable_symbols, closed_symbols)
        """
        tradeable = []
        closed = []

        for symbol in symbols:
            # Get exchange for this symbol
            _, exchange, _ = exchange_manager.parse_symbol(symbol)

            if self.is_trading_day(exchange, check_date):
                tradeable.append(symbol)
            else:
                closed.append(symbol)

        return tradeable, closed

    def get_last_trading_date(self, exchange: str, before_date: date) -> date:
        """
        Get the most recent trading day before a given date.

        Useful for data fetching - ensures we request data from a valid trading day.

        Args:
            exchange: IBKR exchange code
            before_date: Date to start searching backwards from (exclusive)

        Returns:
            Most recent trading date before the given date
        """
        # Start from the day before
        check_date = before_date - timedelta(days=1)

        # Search backwards up to 10 days (handles long holiday periods)
        for _ in range(10):
            if self.is_trading_day(exchange, check_date):
                return check_date
            check_date -= timedelta(days=1)

        # Fallback: just return the original date - 1
        self.logger.warning(
            f"Could not find trading day for {exchange} before {before_date}, "
            f"using {before_date - timedelta(days=1)}"
        )
        return before_date - timedelta(days=1)

    def get_market_status_summary(
        self,
        symbols: List[str],
        check_date: date,
        exchange_manager
    ) -> str:
        """
        Generate a formatted summary of market status for logging.

        Args:
            symbols: List of yfinance symbols
            check_date: Date to check
            exchange_manager: ExchangeManager instance

        Returns:
            Formatted string for logging
        """
        # Group symbols by exchange
        by_exchange: Dict[str, List[str]] = {}
        for symbol in symbols:
            _, exchange, _ = exchange_manager.parse_symbol(symbol)
            if exchange not in by_exchange:
                by_exchange[exchange] = []
            by_exchange[exchange].append(symbol)

        lines = [
            "=" * 60,
            f"MARKET STATUS FOR {check_date}",
            "=" * 60,
        ]

        for exchange, syms in sorted(by_exchange.items()):
            is_open = self.is_trading_day(exchange, check_date)
            status = "OPEN" if is_open else "CLOSED"
            iso_code = self._get_iso_code(exchange)

            # Get holiday name if closed
            holiday_name = ""
            if not is_open and check_date.weekday() not in [5, 6]:
                holiday_name = self._get_holiday_name(exchange, check_date)
                if holiday_name:
                    holiday_name = f" ({holiday_name})"

            lines.append(
                f"  {exchange:6} ({iso_code}): {status:6}{holiday_name} - {len(syms)} symbols"
            )

        lines.append("=" * 60)
        return "\n".join(lines)

    def _get_holiday_name(self, exchange: str, check_date: date) -> str:
        """
        Try to get the holiday name for a closed date.

        Args:
            exchange: IBKR exchange code
            check_date: Date to check

        Returns:
            Holiday name string or empty string
        """
        if not EXCHANGE_CALENDARS_AVAILABLE:
            return ""

        iso_code = self._get_iso_code(exchange)
        calendar = self._get_calendar(iso_code)

        if calendar is None:
            return ""

        try:
            # exchange_calendars doesn't expose holiday names directly
            # Return empty for now - could be enhanced later
            return ""
        except Exception:
            return ""

    def is_open_now(self, exchange: str, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if an exchange is currently open (real-time trading hours check).

        This is different from is_trading_day() which only checks if it's a
        trading day (not weekend/holiday). This method checks actual trading hours.

        Args:
            exchange: IBKR exchange code (e.g., 'TSEJ', 'LSE', 'SMART')
            timestamp: Optional timestamp to check (defaults to now)

        Returns:
            True if the exchange is currently in session, False otherwise
        """
        if not EXCHANGE_CALENDARS_AVAILABLE or not PANDAS_AVAILABLE:
            # Fallback to date-only check if libraries not available
            check_date = timestamp.date() if timestamp else date.today()
            return self.is_trading_day(exchange, check_date)

        # Get current time as pandas Timestamp in UTC
        if timestamp is None:
            now_utc = pd.Timestamp.now(tz='UTC')
        else:
            # Convert datetime to pandas Timestamp with UTC
            if timestamp.tzinfo is None:
                # Assume local time, convert to UTC
                import pytz
                local_tz = pytz.timezone('US/Eastern')  # IBKR default
                timestamp = local_tz.localize(timestamp)
            now_utc = pd.Timestamp(timestamp).tz_convert('UTC')

        iso_code = self._get_iso_code(exchange)
        calendar = self._get_calendar(iso_code)

        if calendar is None:
            # Unknown exchange - fallback to date check
            return self.is_trading_day(exchange, now_utc.date())

        try:
            return calendar.is_open_at_time(now_utc)
        except Exception as e:
            self.logger.warning(f"Error checking real-time status for {exchange}: {e}")
            # Fallback to date check
            return self.is_trading_day(exchange, now_utc.date())

    def get_open_exchanges(self, check_date: date, use_realtime: bool = True) -> List[str]:
        """
        Get list of all open exchanges.

        Args:
            check_date: Date to check (used for fallback and logging)
            use_realtime: If True, check real-time trading hours (default).
                          If False, only check if it's a trading day.

        Returns:
            List of IBKR exchange codes that are open
        """
        open_exchanges = []
        for ibkr_code in self.EXCHANGE_TO_ISO.keys():
            if use_realtime:
                is_open = self.is_open_now(ibkr_code)
            else:
                is_open = self.is_trading_day(ibkr_code, check_date)

            if is_open:
                open_exchanges.append(ibkr_code)
        return open_exchanges

    def get_closed_exchanges(self, check_date: date, use_realtime: bool = True) -> List[str]:
        """
        Get list of all closed exchanges.

        Args:
            check_date: Date to check (used for fallback and logging)
            use_realtime: If True, check real-time trading hours (default).
                          If False, only check if it's a trading day.

        Returns:
            List of IBKR exchange codes that are closed
        """
        closed_exchanges = []
        for ibkr_code in self.EXCHANGE_TO_ISO.keys():
            if use_realtime:
                is_open = self.is_open_now(ibkr_code)
            else:
                is_open = self.is_trading_day(ibkr_code, check_date)

            if not is_open:
                closed_exchanges.append(ibkr_code)
        return closed_exchanges
