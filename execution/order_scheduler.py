"""
Order Scheduler for Per-Exchange Timing

Calculates optimal order submission times based on each exchange's trading hours.
Supports scheduled execution relative to market open, close, or midday.

Trading Hours Reference (all times local to exchange):
- SMART (US NYSE):     09:30-16:00 ET (no lunch break)
- TSEJ (Tokyo):        09:00-15:00 JST (lunch 11:30-12:30)
- LSE (London):        08:00-16:30 GMT (no lunch break)
- SEHK (Hong Kong):    09:30-16:00 HKT (lunch 12:00-13:00)
- BUX (Budapest):      09:00-17:30 CET
- SFB (Stockholm):     09:00-17:30 CET
- BVME (Milan):        09:00-17:30 CET
- SBF (Paris):         09:00-17:30 CET
- IBIS (Frankfurt):    09:00-17:30 CET
- ASX (Sydney):        10:00-16:00 AEST
- TADAWUL (Saudi):     10:00-15:00 AST (Sun-Thu)
- TSE (Toronto):       09:30-16:00 ET
- PRA (Prague):        09:00-17:00 CET
"""

from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple
import logging

try:
    import pytz

    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

try:
    import exchange_calendars as xcals

    EXCHANGE_CALENDARS_AVAILABLE = True
except ImportError:
    EXCHANGE_CALENDARS_AVAILABLE = False


class OrderScheduler:
    """
    Manages order submission timing based on exchange trading hours.

    Calculates when to submit orders for each exchange based on configured
    timing preference (e.g., '30_MIN_AFTER_OPEN', 'MIDDAY').
    """

    # Exchange trading hours: (open_time, close_time, timezone, has_lunch, lunch_start, lunch_end)
    # Times are in 24-hour format as (hour, minute) tuples
    EXCHANGE_HOURS = {
        "SMART": {
            "open": (9, 30),
            "close": (16, 0),
            "timezone": "America/New_York",
            "lunch": None,
        },
        "TSEJ": {
            "open": (9, 0),
            "close": (15, 0),
            "timezone": "Asia/Tokyo",
            "lunch": ((11, 30), (12, 30)),  # Lunch break
        },
        "LSE": {
            "open": (8, 0),
            "close": (16, 30),
            "timezone": "Europe/London",
            "lunch": None,
        },
        "SEHK": {
            "open": (9, 30),
            "close": (16, 0),
            "timezone": "Asia/Hong_Kong",
            "lunch": ((12, 0), (13, 0)),  # Lunch break
        },
        "BUX": {
            "open": (9, 0),
            "close": (17, 30),
            "timezone": "Europe/Budapest",
            "lunch": None,
        },
        "SFB": {
            "open": (9, 0),
            "close": (17, 30),
            "timezone": "Europe/Stockholm",
            "lunch": None,
        },
        "BVME": {
            "open": (9, 0),
            "close": (17, 30),
            "timezone": "Europe/Rome",
            "lunch": None,
        },
        "SBF": {
            "open": (9, 0),
            "close": (17, 30),
            "timezone": "Europe/Paris",
            "lunch": None,
        },
        "IBIS": {
            "open": (9, 0),
            "close": (17, 30),
            "timezone": "Europe/Berlin",
            "lunch": None,
        },
        "ASX": {
            "open": (10, 0),
            "close": (16, 0),
            "timezone": "Australia/Sydney",
            "lunch": None,
        },
        "NSE": {
            "open": (9, 15),
            "close": (15, 30),
            "timezone": "Asia/Kolkata",
            "lunch": None,
        },
        "BSE": {
            "open": (9, 15),
            "close": (15, 30),
            "timezone": "Asia/Kolkata",
            "lunch": None,
        },
        "TADAWUL": {
            "open": (10, 0),
            "close": (15, 0),
            "timezone": "Asia/Riyadh",
            "lunch": None,
        },
        "TSE": {  # Toronto
            "open": (9, 30),
            "close": (16, 0),
            "timezone": "America/Toronto",
            "lunch": None,
        },
        "PRA": {  # Prague
            "open": (9, 0),
            "close": (17, 0),
            "timezone": "Europe/Prague",
            "lunch": None,
        },
        "SGX": {  # Singapore
            "open": (9, 0),
            "close": (17, 0),
            "timezone": "Asia/Singapore",
            "lunch": ((12, 0), (13, 0)),
        },
    }

    # Exchanges that do NOT support All-or-None (AON) orders
    AON_NOT_SUPPORTED = {"TSEJ", "SEHK"}

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize OrderScheduler.

        Args:
            logger: Logger instance for output
        """
        self.logger = logger or logging.getLogger(__name__)

        if not PYTZ_AVAILABLE:
            self.logger.warning(
                "pytz not available - timezone conversions may be inaccurate"
            )

    def get_exchange_hours(self, exchange: str) -> Optional[Dict]:
        """
        Get trading hours for an exchange.

        Args:
            exchange: IBKR exchange code

        Returns:
            Dictionary with open, close, timezone, lunch info or None
        """
        return self.EXCHANGE_HOURS.get(exchange)

    def supports_aon(self, exchange: str) -> bool:
        """
        Check if an exchange supports All-or-None (AON) orders.

        Args:
            exchange: IBKR exchange code

        Returns:
            True if AON is supported, False otherwise
        """
        return exchange not in self.AON_NOT_SUPPORTED

    def _time_to_minutes(self, t: Tuple[int, int]) -> int:
        """Convert (hour, minute) tuple to minutes since midnight."""
        return t[0] * 60 + t[1]

    def _minutes_to_time(self, minutes: int) -> Tuple[int, int]:
        """Convert minutes since midnight to (hour, minute) tuple."""
        return (minutes // 60, minutes % 60)

    def calculate_submission_time(
        self, exchange: str, timing: str, reference_date: Optional[datetime] = None
    ) -> Optional[datetime]:
        """
        Calculate the order submission time for an exchange.

        Args:
            exchange: IBKR exchange code (e.g., 'TSEJ', 'LSE', 'SMART')
            timing: Timing preference from config:
                    'IMMEDIATE', 'AT_OPEN', '30_MIN_AFTER_OPEN', '1_HOUR_AFTER_OPEN',
                    'MIDDAY', '1_HOUR_BEFORE_CLOSE', '30_MIN_BEFORE_CLOSE'
            reference_date: Reference date for calculation (defaults to today)

        Returns:
            datetime object for submission time in UTC, or None for IMMEDIATE
        """
        if timing == "IMMEDIATE":
            return None  # Execute immediately, no scheduling needed

        hours = self.get_exchange_hours(exchange)
        if not hours:
            self.logger.warning(f"Unknown exchange {exchange}, defaulting to IMMEDIATE")
            return None

        # Get reference date
        if reference_date is None:
            reference_date = datetime.now()

        # Calculate target time based on timing preference
        open_minutes = self._time_to_minutes(hours["open"])
        close_minutes = self._time_to_minutes(hours["close"])

        if timing == "AT_OPEN":
            target_minutes = open_minutes

        elif timing == "30_MIN_AFTER_OPEN":
            target_minutes = open_minutes + 30

        elif timing == "1_HOUR_AFTER_OPEN":
            target_minutes = open_minutes + 60

        elif timing == "MIDDAY":
            # For exchanges with lunch break, target after lunch
            if hours["lunch"]:
                lunch_end = hours["lunch"][1]
                target_minutes = (
                    self._time_to_minutes(lunch_end) + 15
                )  # 15 min after lunch
            else:
                # No lunch break - use midpoint of trading session
                target_minutes = (open_minutes + close_minutes) // 2

        elif timing == "1_HOUR_BEFORE_CLOSE":
            target_minutes = close_minutes - 60

        elif timing == "30_MIN_BEFORE_CLOSE":
            target_minutes = close_minutes - 30

        else:
            self.logger.warning(f"Unknown timing {timing}, defaulting to IMMEDIATE")
            return None

        # Ensure target is within trading hours
        target_minutes = max(open_minutes, min(target_minutes, close_minutes - 5))

        # Convert to datetime
        target_hour, target_minute = self._minutes_to_time(target_minutes)

        if PYTZ_AVAILABLE:
            # Create timezone-aware datetime
            tz = pytz.timezone(hours["timezone"])
            local_dt = tz.localize(
                datetime(
                    reference_date.year,
                    reference_date.month,
                    reference_date.day,
                    target_hour,
                    target_minute,
                )
            )
            # Convert to UTC for consistent handling
            utc_dt = local_dt.astimezone(pytz.UTC)
            return utc_dt
        else:
            # Fallback: return naive datetime (local to exchange)
            return datetime(
                reference_date.year,
                reference_date.month,
                reference_date.day,
                target_hour,
                target_minute,
            )

    def calculate_open_plus_minutes(
        self,
        exchange: str,
        minutes_after_open: int,
        reference_date: Optional[datetime] = None,
    ) -> Optional[datetime]:
        """Calculate submission time as market open plus configurable minutes."""
        hours = self.get_exchange_hours(exchange)
        if not hours:
            self.logger.warning(
                f"Unknown exchange {exchange}, cannot calculate open+minutes timing"
            )
            return None

        if reference_date is None:
            reference_date = datetime.now()

        open_minutes = self._time_to_minutes(hours["open"])
        close_minutes = self._time_to_minutes(hours["close"])
        target_minutes = open_minutes + int(minutes_after_open)

        if hours["lunch"]:
            lunch_start = self._time_to_minutes(hours["lunch"][0])
            lunch_end = self._time_to_minutes(hours["lunch"][1])
            if lunch_start <= target_minutes < lunch_end:
                target_minutes = lunch_end + 1

        target_minutes = max(open_minutes, min(target_minutes, close_minutes - 5))
        target_hour, target_minute = self._minutes_to_time(target_minutes)

        if PYTZ_AVAILABLE:
            tz = pytz.timezone(hours["timezone"])
            local_dt = tz.localize(
                datetime(
                    reference_date.year,
                    reference_date.month,
                    reference_date.day,
                    target_hour,
                    target_minute,
                )
            )
            return local_dt.astimezone(pytz.UTC)

        return datetime(
            reference_date.year,
            reference_date.month,
            reference_date.day,
            target_hour,
            target_minute,
        )

    def get_submission_schedule(
        self,
        symbols: list,
        exchange_manager,
        default_timing: str,
        timing_overrides: Optional[Dict[str, str]] = None,
        reference_date: Optional[datetime] = None,
    ) -> Dict[str, Dict]:
        """
        Generate a complete submission schedule for all symbols.

        Args:
            symbols: List of yfinance symbols
            exchange_manager: ExchangeManager instance for symbol->exchange mapping
            default_timing: Default timing from config
            timing_overrides: Per-exchange timing overrides
            reference_date: Reference date for schedule

        Returns:
            Dictionary mapping symbol to schedule info:
            {
                'NVDA': {
                    'exchange': 'SMART',
                    'timing': '30_MIN_AFTER_OPEN',
                    'submission_time': datetime(...),
                    'supports_aon': True,
                },
                ...
            }
        """
        timing_overrides = timing_overrides or {}
        schedule = {}

        for symbol in symbols:
            _, exchange, _ = exchange_manager.parse_symbol(symbol)

            # Determine timing for this exchange
            timing = timing_overrides.get(exchange, default_timing)

            # Calculate submission time
            submission_time = self.calculate_submission_time(
                exchange, timing, reference_date
            )

            schedule[symbol] = {
                "exchange": exchange,
                "timing": timing,
                "submission_time": submission_time,
                "supports_aon": self.supports_aon(exchange),
            }

        return schedule

    def should_submit_now(
        self,
        symbol: str,
        schedule: Dict[str, Dict],
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Check if an order for a symbol should be submitted now.

        Args:
            symbol: Trading symbol
            schedule: Schedule from get_submission_schedule()
            current_time: Current time (defaults to now in UTC)

        Returns:
            Tuple of (should_submit: bool, reason: str)
        """
        if symbol not in schedule:
            return True, "Symbol not in schedule"

        info = schedule[symbol]
        submission_time = info["submission_time"]

        # IMMEDIATE timing - always submit
        if submission_time is None:
            return True, "IMMEDIATE timing"

        # Get current time in UTC
        if current_time is None:
            if PYTZ_AVAILABLE:
                current_time = datetime.now(pytz.UTC)
            else:
                current_time = datetime.utcnow()

        # Make current_time timezone-aware if it isn't
        if PYTZ_AVAILABLE and current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)

        # Check if we've passed the submission time
        if current_time >= submission_time:
            return True, f"Scheduled time reached ({info['timing']})"

        # Calculate wait time
        wait_seconds = (submission_time - current_time).total_seconds()
        wait_minutes = wait_seconds / 60

        return (
            False,
            f"Waiting {wait_minutes:.0f} min until {info['timing']} ({submission_time.strftime('%H:%M %Z')})",
        )

    def get_wait_time_seconds(
        self,
        symbol: str,
        schedule: Dict[str, Dict],
        current_time: Optional[datetime] = None,
    ) -> int:
        """
        Get seconds to wait before submitting an order.

        Args:
            symbol: Trading symbol
            schedule: Schedule from get_submission_schedule()
            current_time: Current time (defaults to now)

        Returns:
            Seconds to wait (0 = submit immediately)
        """
        if symbol not in schedule:
            return 0

        info = schedule[symbol]
        submission_time = info["submission_time"]

        if submission_time is None:
            return 0

        if current_time is None:
            if PYTZ_AVAILABLE:
                current_time = datetime.now(pytz.UTC)
            else:
                current_time = datetime.utcnow()

        if PYTZ_AVAILABLE and current_time.tzinfo is None:
            current_time = pytz.UTC.localize(current_time)

        if current_time >= submission_time:
            return 0

        return int((submission_time - current_time).total_seconds())

    def log_schedule_summary(self, schedule: Dict[str, Dict]) -> str:
        """
        Generate a human-readable schedule summary for logging.

        Args:
            schedule: Schedule from get_submission_schedule()

        Returns:
            Formatted string for logging
        """
        lines = [
            "=" * 60,
            "ORDER SUBMISSION SCHEDULE",
            "=" * 60,
        ]

        # Group by exchange
        by_exchange = {}
        for symbol, info in schedule.items():
            exchange = info["exchange"]
            if exchange not in by_exchange:
                by_exchange[exchange] = []
            by_exchange[exchange].append((symbol, info))

        for exchange in sorted(by_exchange.keys()):
            symbols_info = by_exchange[exchange]
            timing = symbols_info[0][1]["timing"]
            submission_time = symbols_info[0][1]["submission_time"]
            aon_supported = (
                "AON supported" if symbols_info[0][1]["supports_aon"] else "NO AON"
            )

            if submission_time:
                time_str = submission_time.strftime("%H:%M %Z")
            else:
                time_str = "IMMEDIATE"

            symbols = [s for s, _ in symbols_info]
            lines.append(
                f"  {exchange}: {timing} ({time_str}) - {len(symbols)} symbols [{aon_supported}]"
            )
            for sym in symbols:
                lines.append(f"    - {sym}")

        lines.append("=" * 60)
        return "\n".join(lines)
