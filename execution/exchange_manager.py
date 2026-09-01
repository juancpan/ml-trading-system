"""
Exchange Manager for Multi-Exchange Trading

Handles symbol conversion between yfinance format and IBKR format for:
- Tokyo Stock Exchange (TSE): 8002.T -> 8002 on TSE
- London Stock Exchange (LSE): III.L -> III on LSE
- NASDAQ: NVDA -> NVDA on SMART
"""

from typing import Tuple, Optional, Dict
from ibapi.contract import Contract
import logging
from datetime import datetime, time
import pytz


class ExchangeManager:
    """
    Manages exchange-specific symbol conversion and contract creation
    """

    # Exchange mappings (yfinance suffix -> IBKR exchange code)
    # CRITICAL: Every suffix used in config MUST be mapped here!
    # NOTE: For German stocks, use IBIS (Xetra) as it's more reliable than FWB
    EXCHANGE_SUFFIXES = {
        ".T": "TSEJ",  # Tokyo Stock Exchange (Japan)
        ".L": "LSE",  # London Stock Exchange
        ".HK": "SEHK",  # Hong Kong Stock Exchange
        ".AX": "ASX",  # Australian Securities Exchange
        ".SI": "SGX",  # Singapore Exchange (SGX) - D05.SI
        ".PA": "SBF",  # Euronext Paris
        ".DE": "IBIS",  # Deutsche Börse XETRA
        ".F": "FWB2",  # Frankfurt Stock Exchange (Börse Frankfurt) - IBKR uses FWB2, not FWB!
        ".NS": "NSE",  # National Stock Exchange (India)
        ".BO": "BSE",  # Bombay Stock Exchange (India)
        ".MC": "BM",  # Bolsa de Madrid (Spanish Stock Exchange)
        ".TA": "TASE",  # Tel Aviv Stock Exchange (Israel)
        ".TO": "TSE",  # Toronto Stock Exchange (Canada)
        ".OL": "OSE",  # Oslo Stock Exchange (Norway)
        ".SW": "SWX",  # Swiss Exchange (Switzerland) — IBKR official code is SWX
        ".BD": "BUX",  # Budapest Stock Exchange (Hungary)
        ".ST": "SFB",  # Nasdaq Stockholm (Sweden)
        ".SR": "TADAWUL",  # Tadawul (Saudi Stock Exchange)
        ".KL": "MYX",  # Bursa Malaysia (Kuala Lumpur)
        ".PR": "PRA",  # Prague Stock Exchange (Czech Republic)
        ".AE": "DFM",  # Dubai Financial Market (UAE)
        ".AS": "AEB",  # Euronext Amsterdam (Netherlands) — IBKR official code is AEB
        ".MI": "BVME",  # Borsa Italiana Milan (Italy)
        ".CO": "CPH",  # Copenhagen Stock Exchange (Denmark)
        ".RO": "BVB",  # Bucharest Stock Exchange (Romania)
        ".WA": "WSE",  # Warsaw Stock Exchange (Poland)
        ".BR": "ENEXT.BE",  # Euronext Brussels (Belgium) — IBKR code is ENEXT.BE
        ".TL": "N.TALLINN",  # Nasdaq Tallinn (Estonia)
        ".VS": "N.VILNIUS",  # Nasdaq Vilnius (Lithuania)
        ".VI": "VSE",  # Vienna Stock Exchange (Austria)
        ".LS": "BVL",  # Lisbon Stock Exchange (Portugal) - IBKR uses BVL, not LS
        ".SA": "BVMF",  # B3 Brasil Bolsa Balcão (Brazil)
        ".HE": "HEX",  # Nasdaq Helsinki (Finland)
        ".IR": "ISED",  # Irish Stock Exchange (Euronext Dublin)
        ".JO": "JSE",  # Johannesburg Stock Exchange (South Africa)
        ".JK": "IDX",  # Indonesia Stock Exchange
        ".NE": "OMEGA",  # NEO Exchange (Canada)
        ".XA": "ASX",  # ASX alternate suffix (same as .AX)
        ".V": "VENTURE",  # TSX Venture Exchange (Canada)
    }

    # Currency mappings for exchanges
    EXCHANGE_CURRENCIES = {
        "TSEJ": "JPY",  # Tokyo Stock Exchange (Japan)
        "LSE": "GBP",
        "SEHK": "HKD",
        "ASX": "AUD",
        "SGX": "SGD",  # Singapore Exchange
        "SBF": "EUR",
        "IBIS": "EUR",  # Xetra
        "FWB2": "EUR",  # Frankfurt Stock Exchange (IBKR code)
        "NSE": "INR",  # National Stock Exchange (India)
        "BSE": "INR",  # Bombay Stock Exchange (India)
        "BM": "EUR",  # Bolsa de Madrid (Spain)
        "TASE": "ILS",  # Tel Aviv Stock Exchange (Israel)
        "TSE": "CAD",  # Toronto Stock Exchange (Canada)
        "OSE": "NOK",  # Oslo Stock Exchange (Norway)
        "SWX": "CHF",  # Swiss Exchange (Switzerland) — IBKR code SWX
        "BUX": "HUF",  # Budapest Stock Exchange (Hungary)
        "SFB": "SEK",  # Nasdaq Stockholm (Sweden)
        "TADAWUL": "SAR",  # Tadawul (Saudi Arabia)
        "MYX": "MYR",  # Bursa Malaysia
        "PRA": "CZK",  # Prague Stock Exchange (Czech Republic)
        "DFM": "AED",  # Dubai Financial Market (UAE)
        "AEB": "EUR",  # Euronext Amsterdam (Netherlands) — IBKR code AEB
        "BVME": "EUR",  # Borsa Italiana Milan (Italy)
        "CPH": "DKK",  # Copenhagen Stock Exchange (Denmark)
        "BVB": "RON",  # Bucharest Stock Exchange (Romania)
        "WSE": "PLN",  # Warsaw Stock Exchange (Poland)
        "ENEXT.BE": "EUR",  # Euronext Brussels (Belgium)
        "N.TALLINN": "EUR",  # Nasdaq Tallinn (Estonia)
        "N.VILNIUS": "EUR",  # Nasdaq Vilnius (Lithuania)
        "VSE": "EUR",  # Vienna Stock Exchange (Austria)
        "BVL": "EUR",  # Lisbon Stock Exchange (Portugal)
        "BVMF": "BRL",  # B3 Brasil Bolsa Balcão (Brazil)
        "HEX": "EUR",  # Nasdaq Helsinki (Finland)
        "ISED": "EUR",  # Irish Stock Exchange (Euronext Dublin)
        "JSE": "ZAR",  # Johannesburg Stock Exchange (South Africa)
        "IDX": "IDR",  # Indonesia Stock Exchange
        "OMEGA": "CAD",  # NEO Exchange (Canada)
        "VENTURE": "CAD",  # TSX Venture Exchange (Canada)
        "SMART": "USD",  # Default for US stocks
    }

    # Lot size requirements for exchanges
    # Source: Official exchange documentation
    # NOTE: SGX announced reduction from 100 to 10 for stocks >$10 (Nov 2024)
    #       Update this when the change takes effect!
    LOT_SIZES = {
        "TSEJ": 100,  # Tokyo Stock Exchange: 1 lot = 100 shares
        "SEHK": 100,  # Hong Kong: Default 100 shares (varies by stock, see SYMBOL_LOT_SIZES)
        "LSE": 1,  # London: No lot size requirement
        "ASX": 1,  # Australia: No lot size requirement
        "SGX": 100,  # Singapore: 100 shares per lot (board lot) - will change to 10!
        "SBF": 1,  # Euronext: No lot size requirement
        "IBIS": 1,  # Germany XETRA: No lot size requirement
        "FWB2": 1,  # Frankfurt: No lot size requirement
        "NSE": 1,  # India NSE: No lot requirement (cash market)
        "BSE": 1,  # India BSE: No lot requirement (cash market)
        "BM": 1,  # Bolsa de Madrid: No lot size requirement
        "TASE": 1,  # Tel Aviv: No lot size requirement
        "TSE": 1,  # Toronto: No lot size requirement
        "OSE": 1,  # Oslo: No lot size requirement
        "SWX": 1,  # Swiss: No lot size requirement
        "BUX": 1,  # Budapest: No lot size requirement
        "SFB": 1,  # Stockholm: No lot size requirement
        "TADAWUL": 1,  # Tadawul: No lot size requirement
        "MYX": 100,  # Bursa Malaysia: 100 shares per lot
        "PRA": 1,  # Prague: No lot size requirement
        "DFM": 1,  # Dubai: No lot size requirement
        "AEB": 1,  # Amsterdam: No lot size requirement
        "BVME": 1,  # Milan: No lot size requirement
        "CPH": 1,  # Copenhagen: No lot size requirement
        "BVB": 1,  # Bucharest: No lot size requirement
        "WSE": 1,  # Warsaw: No lot size requirement
        "ENEXT.BE": 1,  # Brussels: No lot size requirement
        "N.TALLINN": 1,  # Tallinn: No lot size requirement
        "N.VILNIUS": 1,  # Vilnius: No lot size requirement
        "VSE": 1,  # Vienna: No lot size requirement
        "BVL": 1,  # Lisbon: No lot size requirement
        "SMART": 1,  # US: No lot size requirement
    }

    # Per-symbol lot size overrides (for stocks with non-standard lot sizes)
    # Hong Kong stocks have varying lot sizes per stock
    # Source: HKEX official specifications
    SYMBOL_LOT_SIZES = {
        "1277.HK": 2000,  # Jiangxi Copper: 2000 shares per lot
        "1288.HK": 1000,  # Agricultural Bank of China: 1000 shares per lot
    }

    # Special symbol overrides for IBKR-specific symbols that don't follow standard suffix stripping
    # Format: 'yfinance_symbol': ('ibkr_symbol', 'exchange', 'currency')
    # Use this for symbols where IBKR requires a specific format different from standard parsing
    #
    # IMPORTANT: Rolls-Royce (RR.L) and BAE Systems (BA.L) require special handling because:
    # - Standard parsing would strip .L and send 'RR' or 'BA' to IBKR
    # - IBKR actually expects 'RR.' and 'BA.' (with dot) as the symbol for these LSE stocks
    # - This distinguishes them from US stocks (Boeing is 'BA' without dot)
    SYMBOL_OVERRIDES = {
        # LSE special symbols (IBKR requires trailing dot to disambiguate from US)
        "RR.L": ("RR.", "LSE", "GBP"),  # Rolls-Royce Holdings
        "BA.L": ("BA.", "LSE", "GBP"),  # BAE Systems (not Boeing)
        # LSE IOB depositary receipts — yfinance uses 0xxx.L format, trade on primary exchange
        "0QQ6.L": (
            "ROG",
            "SWX",
            "CHF",
        ),  # Roche Holdings → SIX Swiss primary (IBKR code: SWX)
        # Frankfurt cross-listings — yfinance uses numeric codes, map to primary exchange
        "HI91.F": ("HLT", "NYSE", "USD"),  # Hilton Worldwide → NYSE primary
        "8JO1.F": ("VOLV.B", "SFB", "SEK"),  # Volvo → Stockholm primary
        "4BV.F": ("BVI", "SBF", "EUR"),  # Bureau Veritas → Euronext Paris primary
        "6EQ.F": ("EQT", "SFB", "SEK"),  # EQT AB → Stockholm primary
        # Milan cross-listings of German stocks — yfinance prefixes with number
        "1BAYN.MI": ("BAYN", "IBIS", "EUR"),  # Bayer → Xetra primary
        "1MRK.MI": ("MRK", "IBIS", "EUR"),  # Merck KGaA → Xetra primary
        # Samsung SDR on Vienna (yfinance uses SSU.VI)
        "SSU.VI": ("SMSN", "LSE", "GBP"),  # Samsung SDR → trade via LSE IOB
    }

    # Reverse mapping: IBKR symbol -> yfinance symbol (for position loading from IBKR)
    # Used when IBKR returns positions and we need to match to yfinance format
    IBKR_TO_YFINANCE_MAP = {
        "RR.": "RR.L",  # IBKR returns 'RR.' for Rolls-Royce
        "BA.": "BA.L",  # IBKR returns 'BA.' for BAE Systems
        "ROG": "0QQ6.L",  # Roche on SIX → yfinance IOB code
        "HLT": "HI91.F",  # Hilton
        "BVI": "4BV.F",  # Bureau Veritas
    }

    # Market close times (local exchange time)
    MARKET_CLOSE_TIMES = {
        "TSEJ": {"hour": 15, "minute": 0, "timezone": "Asia/Tokyo"},  # 15:00 JST
        "LSE": {"hour": 16, "minute": 30, "timezone": "Europe/London"},  # 16:30 GMT
        "NSE": {"hour": 15, "minute": 30, "timezone": "Asia/Kolkata"},  # 15:30 IST
        "BSE": {"hour": 15, "minute": 30, "timezone": "Asia/Kolkata"},  # 15:30 IST
        "SEHK": {"hour": 16, "minute": 0, "timezone": "Asia/Hong_Kong"},  # 16:00 HKT
        "ASX": {"hour": 16, "minute": 0, "timezone": "Australia/Sydney"},  # 16:00 AEDT
        "SGX": {"hour": 17, "minute": 0, "timezone": "Asia/Singapore"},  # 17:00 SGT
        "SBF": {"hour": 17, "minute": 30, "timezone": "Europe/Paris"},  # 17:30 CET
        "IBIS": {"hour": 17, "minute": 30, "timezone": "Europe/Berlin"},  # 17:30 CET
        "FWB2": {
            "hour": 20,
            "minute": 0,
            "timezone": "Europe/Berlin",
        },  # 20:00 CET (extended hours)
        "BM": {
            "hour": 17,
            "minute": 30,
            "timezone": "Europe/Madrid",
        },  # 17:30 CET (Bolsa de Madrid)
        "SMART": {
            "hour": 16,
            "minute": 0,
            "timezone": "America/New_York",
        },  # 16:00 ET (4 PM)
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize exchange manager

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def parse_symbol(self, yfinance_symbol: str) -> Tuple[str, str, str]:
        """
        Parse yfinance symbol to extract IBKR symbol, exchange, and currency

        Args:
            yfinance_symbol: Symbol in yfinance format (e.g., '8002.T', 'III.L', 'NVDA')

        Returns:
            Tuple of (ibkr_symbol, exchange, currency)

        Examples:
            >>> parse_symbol('8002.T')
            ('8002', 'TSE', 'JPY')

            >>> parse_symbol('III.L')
            ('III', 'LSE', 'GBP')

            >>> parse_symbol('NVDA')
            ('NVDA', 'SMART', 'USD')

            >>> parse_symbol('RR.')
            ('RR.', 'LSE', 'GBP')  # Special IBKR symbol override
        """
        # Check for special symbol overrides first (IBKR-specific symbols)
        if yfinance_symbol in self.SYMBOL_OVERRIDES:
            ibkr_symbol, exchange, currency = self.SYMBOL_OVERRIDES[yfinance_symbol]
            self.logger.debug(
                f"Parsed {yfinance_symbol} -> symbol={ibkr_symbol}, exchange={exchange}, currency={currency} (override)"
            )
            return ibkr_symbol, exchange, currency

        # Check for exchange suffix
        for suffix, exchange in self.EXCHANGE_SUFFIXES.items():
            if yfinance_symbol.endswith(suffix):
                # Remove suffix to get IBKR symbol
                ibkr_symbol = yfinance_symbol[: -len(suffix)]
                currency = self.EXCHANGE_CURRENCIES[exchange]

                # Nordic share classes: convert hyphen to dot
                # yfinance uses 'SAAB-B.ST', 'NOVO-B.CO', 'NDA-FI.HE'
                # IBKR expects 'SAAB.B', 'NOVO.B', 'NDA.FI'
                if exchange in ("SFB", "CPH", "HEX", "OSE") and "-" in ibkr_symbol:
                    ibkr_symbol = ibkr_symbol.replace("-", ".")
                    self.logger.debug(
                        f"Nordic share class conversion: {yfinance_symbol} -> {ibkr_symbol}"
                    )

                # Canadian unit trusts and share classes: convert hyphen to dot
                # yfinance uses 'CSH-UN.TO', 'BBD-B.TO'
                # IBKR expects 'CSH.UN', 'BBD.B'
                if exchange in ("TSE", "OMEGA", "VENTURE") and "-" in ibkr_symbol:
                    ibkr_symbol = ibkr_symbol.replace("-", ".")
                    self.logger.debug(
                        f"Canadian symbol conversion: {yfinance_symbol} -> {ibkr_symbol}"
                    )

                # UK share classes: convert hyphen to dot
                # yfinance uses 'BT-A.L', IBKR expects 'BT.A'
                if exchange == "LSE" and "-" in ibkr_symbol:
                    ibkr_symbol = ibkr_symbol.replace("-", ".")
                    self.logger.debug(
                        f"UK share class conversion: {yfinance_symbol} -> {ibkr_symbol}"
                    )

                # Hong Kong (SEHK): strip leading zeros
                # yfinance uses '0700.HK', IBKR expects '700'
                if exchange == "SEHK" and ibkr_symbol.lstrip("0"):
                    stripped = ibkr_symbol.lstrip("0")
                    if stripped != ibkr_symbol:
                        self.logger.debug(
                            f"SEHK leading-zero strip: {ibkr_symbol} -> {stripped}"
                        )
                        ibkr_symbol = stripped

                self.logger.debug(
                    f"Parsed {yfinance_symbol} -> symbol={ibkr_symbol}, exchange={exchange}, currency={currency}"
                )
                return ibkr_symbol, exchange, currency

        # Default to US market (SMART routing)
        ibkr_symbol = yfinance_symbol

        # US share classes: convert hyphen to space
        # yfinance uses 'BRK-B', IBKR expects 'BRK B'
        if "-" in ibkr_symbol:
            ibkr_symbol = ibkr_symbol.replace("-", " ")
            self.logger.debug(
                f"US share class conversion: {yfinance_symbol} -> {ibkr_symbol}"
            )

        self.logger.debug(
            f"Parsed {yfinance_symbol} -> symbol={ibkr_symbol}, exchange=SMART, currency=USD"
        )
        return ibkr_symbol, "SMART", "USD"

    def to_yfinance_symbol(self, ibkr_symbol: str, exchange: str) -> str:
        """
        Convert IBKR symbol and exchange to yfinance format

        Args:
            ibkr_symbol: IBKR ticker symbol
            exchange: Exchange code (TSE, LSE, SMART, etc.)

        Returns:
            yfinance symbol with appropriate suffix

        Examples:
            >>> to_yfinance_symbol('8002', 'TSE')
            '8002.T'

            >>> to_yfinance_symbol('III', 'LSE')
            'III.L'

            >>> to_yfinance_symbol('NVDA', 'SMART')
            'NVDA'
        """
        # Find suffix for exchange
        for suffix, exch in self.EXCHANGE_SUFFIXES.items():
            if exch == exchange:
                # Reverse share class conversions (dot back to hyphen)
                symbol_for_yf = ibkr_symbol

                # Nordic: IBKR 'SAAB.B' -> yfinance 'SAAB-B'
                if exchange in ("SFB", "CPH", "HEX", "OSE") and "." in ibkr_symbol:
                    symbol_for_yf = ibkr_symbol.replace(".", "-")
                    self.logger.debug(
                        f"Nordic reverse conversion: {ibkr_symbol} -> {symbol_for_yf}"
                    )

                # Canadian: IBKR 'CSH.UN' -> yfinance 'CSH-UN'
                if exchange in ("TSE", "OMEGA", "VENTURE") and "." in ibkr_symbol:
                    symbol_for_yf = ibkr_symbol.replace(".", "-")
                    self.logger.debug(
                        f"Canadian reverse conversion: {ibkr_symbol} -> {symbol_for_yf}"
                    )

                # UK: IBKR 'BT.A' -> yfinance 'BT-A'
                if exchange == "LSE" and "." in ibkr_symbol:
                    symbol_for_yf = ibkr_symbol.replace(".", "-")
                    self.logger.debug(
                        f"UK reverse conversion: {ibkr_symbol} -> {symbol_for_yf}"
                    )

                yf_symbol = f"{symbol_for_yf}{suffix}"
                self.logger.debug(
                    f"Converted {ibkr_symbol} on {exchange} -> {yf_symbol}"
                )
                return yf_symbol

        # No suffix needed for SMART routing (US stocks)
        # Reverse US share class: IBKR 'BRK B' -> yfinance 'BRK-B'
        yf_symbol = ibkr_symbol
        if " " in ibkr_symbol:
            yf_symbol = ibkr_symbol.replace(" ", "-")
            self.logger.debug(f"US reverse conversion: {ibkr_symbol} -> {yf_symbol}")
        self.logger.debug(f"Converted {ibkr_symbol} on {exchange} -> {yf_symbol}")
        return yf_symbol

    def create_contract(self, yfinance_symbol: str) -> Contract:
        """
        Create IBKR Stock contract from yfinance symbol using official IBKR API.

        Follows official IBKR API specification from:
        https://interactivebrokers.github.io/tws-api/basic_contracts.html

        Args:
            yfinance_symbol: Symbol in yfinance format (e.g., '8002.T', 'III.L', 'NVDA')

        Returns:
            ibapi.contract.Contract object configured for the appropriate exchange
        """
        ibkr_symbol, exchange, currency = self.parse_symbol(yfinance_symbol)

        # Official IBKR API method for creating stock contracts
        contract = Contract()
        contract.symbol = ibkr_symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency

        self.logger.info(f"Created contract: {ibkr_symbol} on {exchange} ({currency})")
        return contract

    def get_currency(self, yfinance_symbol: str) -> str:
        """
        Get currency for a given yfinance symbol

        Args:
            yfinance_symbol: Symbol in yfinance format

        Returns:
            Currency code (USD, JPY, GBP, etc.)
        """
        _, _, currency = self.parse_symbol(yfinance_symbol)
        return currency

    def get_yfinance_symbol(self, config_symbol: str) -> str:
        """
        Convert config symbol to yfinance format for data fetching.

        In our current design, config.py uses yfinance format (e.g., 'RR.L', 'BA.L'),
        so this method is mostly a pass-through. It handles the reverse case where
        IBKR position data needs to be converted back to yfinance format.

        Args:
            config_symbol: Symbol as used in config.py (e.g., 'RR.L', 'NVDA', '8002.T')

        Returns:
            yfinance-compatible symbol (same as input for config symbols)
        """
        # Check IBKR_TO_YFINANCE_MAP for reverse conversion (IBKR -> yfinance)
        # This handles cases where IBKR returns 'RR.' and we need 'RR.L'
        if config_symbol in self.IBKR_TO_YFINANCE_MAP:
            yf_symbol = self.IBKR_TO_YFINANCE_MAP[config_symbol]
            self.logger.debug(
                f"Converted IBKR symbol {config_symbol} -> {yf_symbol} for yfinance"
            )
            return yf_symbol

        # Config symbols already use yfinance format
        return config_symbol

    def ibkr_to_yfinance_symbol(self, ibkr_symbol: str, exchange: str) -> str:
        """
        Convert IBKR symbol back to yfinance format.

        Used when loading positions from IBKR to match with config symbols.

        Args:
            ibkr_symbol: Symbol as returned by IBKR (e.g., 'RR.', '8002', 'NVDA')
            exchange: Exchange code from IBKR

        Returns:
            yfinance-compatible symbol (e.g., 'RR.L', '8002.T', 'NVDA')
        """
        # Check for special IBKR symbols first
        if ibkr_symbol in self.IBKR_TO_YFINANCE_MAP:
            yf_symbol = self.IBKR_TO_YFINANCE_MAP[ibkr_symbol]
            self.logger.debug(f"Converted IBKR {ibkr_symbol} -> {yf_symbol}")
            return yf_symbol

        # Standard conversion: add suffix based on exchange
        return self.to_yfinance_symbol(ibkr_symbol, exchange)

    def get_exchange(self, yfinance_symbol: str) -> str:
        """
        Get exchange for a given yfinance symbol

        Args:
            yfinance_symbol: Symbol in yfinance format

        Returns:
            Exchange code (TSE, LSE, SMART, etc.)
        """
        _, exchange, _ = self.parse_symbol(yfinance_symbol)
        return exchange

    def validate_symbol(self, yfinance_symbol: str) -> bool:
        """
        Validate that a symbol can be parsed correctly

        Args:
            yfinance_symbol: Symbol in yfinance format

        Returns:
            True if valid, False otherwise
        """
        try:
            ibkr_symbol, exchange, currency = self.parse_symbol(yfinance_symbol)

            # Basic validation
            if not ibkr_symbol or not exchange or not currency:
                return False

            # Ensure currency is valid for exchange
            expected_currency = self.EXCHANGE_CURRENCIES.get(exchange)
            if expected_currency and expected_currency != currency:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating symbol {yfinance_symbol}: {e}")
            return False

    def round_to_lot_size(self, yfinance_symbol: str, quantity: float) -> int:
        """
        Round quantity to exchange-specific lot size requirements.

        Tokyo Stock Exchange requires multiples of 100 shares.
        Hong Kong stocks: Round to nearest 1000 (compatible with lot sizes of 1000 and 2000).
        Other exchanges may have different requirements.

        Args:
            yfinance_symbol: Symbol in yfinance format (e.g., '8002.T', '1277.HK')
            quantity: Desired quantity (can be float)

        Returns:
            Rounded quantity as integer, compliant with exchange rules

        Examples:
            >>> round_to_lot_size('8002.T', 7213)
            7200  # Tokyo: Rounded to nearest 100

            >>> round_to_lot_size('1277.HK', 6070)
            6000  # HK: Rounded to nearest 1000

            >>> round_to_lot_size('1288.HK', 2119)
            2000  # HK: Rounded to nearest 1000

            >>> round_to_lot_size('NVDA', 255.7)
            256  # US stocks: round to nearest integer
        """
        _, exchange, _ = self.parse_symbol(yfinance_symbol)

        # Special handling for Hong Kong stocks
        # HKEX stocks have varying lot sizes (1000, 2000, etc.)
        # Round to nearest 1000 to ensure compatibility
        if exchange == "SEHK":
            # Round to nearest 1000 for Hong Kong stocks
            rounded = int(round(quantity / 1000) * 1000)
            if rounded != int(quantity):
                # Get actual lot size if defined
                actual_lot_size = self.SYMBOL_LOT_SIZES.get(yfinance_symbol, 1000)
                self.logger.info(
                    f"Rounded {yfinance_symbol} quantity: {quantity} -> {rounded} "
                    f"(rounded to nearest 1000, actual lot size: {actual_lot_size})"
                )
            return rounded

        # Standard lot size handling for other exchanges
        lot_size = self.LOT_SIZES.get(exchange, 1)

        if lot_size == 1:
            # No lot size requirement, just round to integer
            rounded = int(round(quantity))
        else:
            # Round to nearest multiple of lot_size
            rounded = int(round(quantity / lot_size) * lot_size)

        if rounded != int(quantity):
            self.logger.info(
                f"Rounded {yfinance_symbol} quantity: {quantity} -> {rounded} (lot size: {lot_size})"
            )

        return rounded

    def get_lot_size(self, yfinance_symbol: str) -> int:
        """
        Get the lot size requirement for a symbol.

        Args:
            yfinance_symbol: Symbol in yfinance format

        Returns:
            Lot size (e.g., 100 for Tokyo, 1000 for 1288.HK, 2000 for 1277.HK)
        """
        # Check for per-symbol override first
        if yfinance_symbol in self.SYMBOL_LOT_SIZES:
            return self.SYMBOL_LOT_SIZES[yfinance_symbol]

        # Fall back to exchange-level lot size
        _, exchange, _ = self.parse_symbol(yfinance_symbol)
        return self.LOT_SIZES.get(exchange, 1)

    def get_tick_size(self, yfinance_symbol: str, price: float) -> float:
        """
        Get minimum price variation (tick size) for a symbol.
        Critical for Tokyo stocks - TOPIX 100 uses ¥0.5 ticks, not ¥0.01.
        Prevents IBKR error 110: "price does not conform to minimum price variation"

        Args:
            yfinance_symbol: Symbol in yfinance format
            price: Current price in native currency

        Returns:
            Tick size in native currency
        """
        _, exchange, _ = self.parse_symbol(yfinance_symbol)

        # Tokyo Stock Exchange - TOPIX 100/500 use decimal pricing
        if exchange == "TSEJ":
            # Marubeni (8002.T) is TOPIX 100 constituent
            # TOPIX 100/500 stocks use ¥0.5 tick in ¥1,000-5,000 range
            if 1000 <= price <= 5000:
                return 0.5  # ¥0.5 tick (e.g., 3758.0, 3758.5, 3759.0)
            elif price < 1000:
                return 0.1  # ¥0.1 tick for lower prices
            else:
                return 1.0  # ¥1 tick for higher prices

        # NSE/BSE India - ₹0.01 tick
        elif exchange in ["NSE", "BSE"]:
            return 0.01

        # LSE - £0.01 tick (pence)
        elif exchange == "LSE":
            return 0.01

        # Other exchanges - $0.01 default
        else:
            return 0.01

    def market_has_closed_today(self, yfinance_symbol: str) -> bool:
        """
        Check if the market for this symbol has closed for today.
        Used to determine whether to fetch today's or yesterday's data.

        Args:
            yfinance_symbol: Symbol in yfinance format

        Returns:
            bool: True if market has closed, False if still open or unknown
        """
        _, exchange, _ = self.parse_symbol(yfinance_symbol)

        if exchange not in self.MARKET_CLOSE_TIMES:
            # Unknown exchange - assume US market hours as default
            self.logger.debug(f"Unknown market hours for {exchange}, assuming closed")
            return True

        market_info = self.MARKET_CLOSE_TIMES[exchange]

        try:
            # Get current time in market's timezone
            market_tz = pytz.timezone(market_info["timezone"])
            current_time_local = datetime.now(market_tz)

            # Create market close time for today
            close_time = current_time_local.replace(
                hour=market_info["hour"],
                minute=market_info["minute"],
                second=0,
                microsecond=0,
            )

            # Check if current time is past market close
            has_closed = current_time_local > close_time

            self.logger.debug(
                f"{yfinance_symbol} ({exchange}): Current={current_time_local.strftime('%H:%M %Z')}, "
                f"Close={close_time.strftime('%H:%M %Z')}, Has closed: {has_closed}"
            )

            return has_closed

        except Exception as e:
            self.logger.warning(
                f"Error checking market hours for {yfinance_symbol}: {e}, assuming closed"
            )
            return True  # Safe default: assume closed

    def get_market_calendar(self):
        """
        Lazy-load MarketCalendarManager for per-exchange holiday detection.

        Returns:
            MarketCalendarManager instance
        """
        if not hasattr(self, "_market_calendar"):
            from market_calendar import MarketCalendarManager

            self._market_calendar = MarketCalendarManager(self.logger)
        return self._market_calendar

    def test_symbol_conversions(self) -> Dict[str, bool]:
        """
        Test symbol conversions for all supported exchanges.

        Returns:
            Dict mapping test name to pass/fail status
        """
        from typing import Dict

        test_results = {}

        # Test cases: (yfinance_symbol, expected_ibkr_symbol, expected_exchange, expected_currency)
        test_cases = [
            ("8002.T", "8002", "TSEJ", "JPY"),  # Tokyo
            ("8001.T", "8001", "TSEJ", "JPY"),  # Tokyo (Itochu)
            ("III.L", "III", "LSE", "GBP"),  # London (standard .L suffix)
            (
                "RR.L",
                "RR.",
                "LSE",
                "GBP",
            ),  # London (Rolls-Royce) - SYMBOL_OVERRIDE: RR.L -> RR.
            (
                "BA.L",
                "BA.",
                "LSE",
                "GBP",
            ),  # London (BAE Systems) - SYMBOL_OVERRIDE: BA.L -> BA.
            ("TATASTEEL.NS", "TATASTEEL", "NSE", "INR"),  # India NSE
            ("RELIANCE.NS", "RELIANCE", "NSE", "INR"),  # India NSE
            ("TATASTEEL.BO", "TATASTEEL", "BSE", "INR"),  # India BSE
            ("0700.HK", "0700", "SEHK", "HKD"),  # Hong Kong
            ("1919.HK", "1919", "SEHK", "HKD"),  # Hong Kong (CSCI)
            ("BHP.AX", "BHP", "ASX", "AUD"),  # Australia
            ("SLX.AX", "SLX", "ASX", "AUD"),  # Australia (Silex)
            ("D05.SI", "D05", "SGX", "SGD"),  # Singapore (DBS)
            ("AIR.PA", "AIR", "SBF", "EUR"),  # Paris
            ("BMW.DE", "BMW", "IBIS", "EUR"),  # Germany XETRA
            ("CEZ.F", "CEZ", "FWB2", "EUR"),  # Frankfurt (CEZ) - IBKR uses FWB2
            ("ITX.MC", "ITX", "BM", "EUR"),  # Madrid (Inditex)
            ("NVDA", "NVDA", "SMART", "USD"),  # US
            ("AAPL", "AAPL", "SMART", "USD"),  # US
            (
                "SAAB-B.ST",
                "SAAB.B",
                "SFB",
                "SEK",
            ),  # Sweden (SAAB B-shares) - hyphen to dot
            (
                "VOLV-B.ST",
                "VOLV.B",
                "SFB",
                "SEK",
            ),  # Sweden (Volvo B-shares) - hyphen to dot
            ("ABB.ST", "ABB", "SFB", "SEK"),  # Sweden (no share class) - unchanged
            (
                "CSH-UN.TO",
                "CSH.UN",
                "TSE",
                "CAD",
            ),  # Canada (unit trust) - hyphen to dot
            ("RST.TO", "RST", "TSE", "CAD"),  # Canada (regular stock) - unchanged
        ]

        for yf_symbol, exp_ibkr, exp_exchange, exp_currency in test_cases:
            test_name = f"parse_{yf_symbol}"
            try:
                ibkr_symbol, exchange, currency = self.parse_symbol(yf_symbol)

                if (
                    ibkr_symbol == exp_ibkr
                    and exchange == exp_exchange
                    and currency == exp_currency
                ):
                    test_results[test_name] = True
                    self.logger.info(
                        f"✓ {test_name}: {yf_symbol} -> {ibkr_symbol} on {exchange} ({currency})"
                    )
                else:
                    test_results[test_name] = False
                    self.logger.error(
                        f"✗ {test_name}: Expected {exp_ibkr}/{exp_exchange}/{exp_currency}, "
                        f"got {ibkr_symbol}/{exchange}/{currency}"
                    )

            except Exception as e:
                test_results[test_name] = False
                self.logger.error(f"✗ {test_name}: Exception - {e}")

        # Test reverse conversions (IBKR -> yfinance)
        # Standard conversions using to_yfinance_symbol()
        reverse_cases = [
            ("8002", "TSEJ", "8002.T"),
            ("III", "LSE", "III.L"),
            ("TATASTEEL", "NSE", "TATASTEEL.NS"),
            ("BHP", "ASX", "BHP.AX"),
            ("D05", "SGX", "D05.SI"),
            ("CEZ", "FWB2", "CEZ.F"),
            ("NVDA", "SMART", "NVDA"),
            ("SAAB.B", "SFB", "SAAB-B.ST"),  # Sweden (SAAB B-shares) - dot to hyphen
            ("VOLV.B", "SFB", "VOLV-B.ST"),  # Sweden (Volvo B-shares) - dot to hyphen
            ("ABB", "SFB", "ABB.ST"),  # Sweden (no share class) - unchanged
            ("CSH.UN", "TSE", "CSH-UN.TO"),  # Canada (unit trust) - dot to hyphen
            ("RST", "TSE", "RST.TO"),  # Canada (regular stock) - unchanged
        ]

        # Test special IBKR symbol -> yfinance conversions using ibkr_to_yfinance_symbol()
        # These use IBKR_TO_YFINANCE_MAP for symbols like RR. -> RR.L
        special_reverse_cases = [
            ("RR.", "LSE", "RR.L"),  # Rolls-Royce: IBKR returns RR., we need RR.L
            ("BA.", "LSE", "BA.L"),  # BAE Systems: IBKR returns BA., we need BA.L
        ]

        for ibkr_symbol, exchange, exp_yf in special_reverse_cases:
            test_name = f"ibkr_reverse_{ibkr_symbol}_{exchange}"
            try:
                yf_symbol = self.ibkr_to_yfinance_symbol(ibkr_symbol, exchange)

                if yf_symbol == exp_yf:
                    test_results[test_name] = True
                    self.logger.info(
                        f"✓ {test_name}: {ibkr_symbol}/{exchange} -> {yf_symbol}"
                    )
                else:
                    test_results[test_name] = False
                    self.logger.error(
                        f"✗ {test_name}: Expected {exp_yf}, got {yf_symbol}"
                    )

            except Exception as e:
                test_results[test_name] = False
                self.logger.error(f"✗ {test_name}: Exception - {e}")

        for ibkr_symbol, exchange, exp_yf in reverse_cases:
            test_name = f"reverse_{ibkr_symbol}_{exchange}"
            try:
                yf_symbol = self.to_yfinance_symbol(ibkr_symbol, exchange)

                if yf_symbol == exp_yf:
                    test_results[test_name] = True
                    self.logger.info(
                        f"✓ {test_name}: {ibkr_symbol}/{exchange} -> {yf_symbol}"
                    )
                else:
                    test_results[test_name] = False
                    self.logger.error(
                        f"✗ {test_name}: Expected {exp_yf}, got {yf_symbol}"
                    )

            except Exception as e:
                test_results[test_name] = False
                self.logger.error(f"✗ {test_name}: Exception - {e}")

        return test_results

    def validate_config_symbols(self, symbols: list = None) -> Dict[str, Dict]:
        """
        CRITICAL PRE-TRADE VALIDATION: Validate ALL symbols in config have correct mappings.

        This should be run BEFORE live trading to ensure:
        1. All symbols have valid exchange mappings
        2. No symbols will default to SMART (US) incorrectly
        3. Currencies are correct for each exchange

        Args:
            symbols: List of yfinance symbols to validate (defaults to SYMBOLS from config)

        Returns:
            Dict with validation results per symbol:
            {
                'symbol': {
                    'valid': bool,
                    'ibkr_symbol': str,
                    'exchange': str,
                    'currency': str,
                    'lot_size': int,
                    'warning': str or None,
                    'error': str or None
                }
            }
        """
        if symbols is None:
            from config import SYMBOLS

            symbols = SYMBOLS

        results = {}
        errors = []
        warnings = []

        self.logger.info("=" * 70)
        self.logger.info("PRE-TRADE SYMBOL VALIDATION")
        self.logger.info("=" * 70)
        self.logger.info(
            f"{'Symbol':<15} {'IBKR Symbol':<12} {'Exchange':<8} {'Currency':<8} {'Lot':<6} Status"
        )
        self.logger.info("-" * 70)

        for yf_symbol in symbols:
            result = {
                "valid": True,
                "ibkr_symbol": None,
                "exchange": None,
                "currency": None,
                "lot_size": None,
                "warning": None,
                "error": None,
            }

            try:
                ibkr_symbol, exchange, currency = self.parse_symbol(yf_symbol)
                lot_size = self.get_lot_size(yf_symbol)

                result["ibkr_symbol"] = ibkr_symbol
                result["exchange"] = exchange
                result["currency"] = currency
                result["lot_size"] = lot_size

                # Check for potential issues
                status = "✓"

                # Warning: Symbol has suffix but mapped to SMART (US)
                if "." in yf_symbol and exchange == "SMART":
                    result["warning"] = (
                        f"Symbol has suffix but mapped to SMART - possible missing mapping!"
                    )
                    result["valid"] = False
                    warnings.append(yf_symbol)
                    status = "⚠️ UNMAPPED SUFFIX"

                # Warning: Non-US exchange but USD currency
                if exchange != "SMART" and currency == "USD":
                    result["warning"] = (
                        f"Non-US exchange {exchange} but USD currency - verify!"
                    )
                    warnings.append(yf_symbol)
                    status = "⚠️ CURRENCY MISMATCH"

                # Check if exchange is in lot sizes (should always be)
                if exchange not in self.LOT_SIZES:
                    result["warning"] = f"Exchange {exchange} not in LOT_SIZES"
                    warnings.append(yf_symbol)

                self.logger.info(
                    f"{yf_symbol:<15} {ibkr_symbol:<12} {exchange:<8} {currency:<8} {lot_size:<6} {status}"
                )

            except Exception as e:
                result["valid"] = False
                result["error"] = str(e)
                errors.append(yf_symbol)
                self.logger.error(
                    f"{yf_symbol:<15} {'ERROR':<12} {'':<8} {'':<8} {'':<6} ✗ {e}"
                )

            results[yf_symbol] = result

        self.logger.info("-" * 70)

        # Summary
        valid_count = sum(1 for r in results.values() if r["valid"])
        self.logger.info(f"Valid: {valid_count}/{len(symbols)}")

        if errors:
            self.logger.error(f"❌ ERRORS (will fail): {errors}")

        if warnings:
            self.logger.warning(f"⚠️  WARNINGS (verify): {warnings}")

        if not errors and not warnings:
            self.logger.info("✅ All symbols validated successfully!")

        self.logger.info("=" * 70)

        return results

    def get_unmapped_suffixes(self, symbols: list = None) -> list:
        """
        Find any symbols with suffixes that are NOT properly mapped.

        This catches configuration errors where a new exchange suffix is used
        but not mapped, which would cause it to default to SMART (US).

        Checks both EXCHANGE_SUFFIXES and SYMBOL_OVERRIDES for valid mappings.

        Args:
            symbols: List of symbols to check

        Returns:
            List of (symbol, suffix) tuples for unmapped suffixes
        """
        if symbols is None:
            from config import SYMBOLS

            symbols = SYMBOLS

        unmapped = []

        for symbol in symbols:
            # Skip if symbol has a direct override (e.g., 'RR.', 'BA.')
            if symbol in self.SYMBOL_OVERRIDES:
                continue

            if "." in symbol:
                # Extract suffix (everything from last '.')
                suffix = "." + symbol.split(".")[-1]

                if suffix not in self.EXCHANGE_SUFFIXES:
                    unmapped.append((symbol, suffix))
                    self.logger.error(
                        f"UNMAPPED SUFFIX: {symbol} has suffix '{suffix}' not in EXCHANGE_SUFFIXES!"
                    )

        return unmapped


# Convenience function for quick access
def create_exchange_manager(logger: Optional[logging.Logger] = None) -> ExchangeManager:
    """Create and return an ExchangeManager instance"""
    return ExchangeManager(logger=logger)
