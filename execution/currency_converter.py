"""
Currency Converter for Multi-Currency Position Sizing

Fetches forex rates from IBKR and converts all foreign currencies to USD.
For currencies not available on IBKR (INR, RON), uses external API or IBKR historical fallback.

Follows official IBKR API specification:
https://interactivebrokers.github.io/tws-api/basic_contracts.html (Forex/Cash contracts)
https://interactivebrokers.github.io/tws-api/md_request.html (Market data requests)

Supported Currencies:
- Major (IDEALPRO): USD, GBP, JPY, EUR, HKD, AUD, SGD, CAD, CHF, NZD
- Exotic (IDEALPRO): HUF, SEK, CZK, NOK, PLN, DKK, ILS, MXN, ZAR, TRY
- External API only: INR, RON (not on IDEALPRO)
- Pegged: SAR (fixed 3.75 USD/SAR)

Fallback Strategy (4-level):
1. IBKR IDEALPRO market data (primary)
2. IBKR historical MIDPOINT rate (secondary)
3. Frankfurter.dev API (tertiary - European Central Bank rates)
4. Config fallback rate from CURRENCY_RATE_FALLBACKS (never returns None)
"""

import threading
import time
import json
import random
from pathlib import Path
from typing import Dict, Optional
from ibapi.contract import Contract
import requests


class CurrencyConverter:
    """
    Manages forex rate fetching and currency conversion using IBKR API.

    Official IBKR API Methods Used:
    - reqMktData(tickerId, contract, genericTickList, snapshot, regulatory snapshot, mktDataOptions)
    - tickPrice(tickerId, tickType, price, attrib) - callback
    """

    # Tick types from IBKR API
    TICK_BID = 1
    TICK_ASK = 2
    TICK_LAST = 4
    TICK_CLOSE = 9

    def __init__(self, ib_client, logger):
        """
        Initialize currency converter.

        Args:
            ib_client: IBClient instance with connection to IB Gateway
            logger: Logger instance
        """
        self.ib_client = ib_client
        self.logger = logger

        # Cache for forex rates {currency_pair: rate}
        self.rates = {}

        # Threading for synchronous API calls
        self.rate_events = {}  # {tickerId: threading.Event()}
        self.pending_forex_requests = {}  # {tickerId: currency_pair}

        # Temporary storage for tick data
        self.tick_data = {}  # {tickerId: {bid, ask, last}}

        # Ticker ID management (start from 6000 to avoid conflicts)
        self.current_tickerId = 6000
        self.tickerId_lock = threading.Lock()

        # Persistence
        self.cache_file = Path("forex_rates_cache.json")
        self.last_update = None
        self.update_interval = 3600  # 1 hour
        self.load_cache()

        # External API support for currencies not available on IBKR (e.g., INR)
        self.external_api_cache = {}  # {currency_pair: rate}
        self.external_api_timestamp = {}  # {currency_pair: timestamp}
        self.external_api_duration = 3600  # 1 hour cache

        # Persistence for external API rates (different from IBKR forex cache)
        self.external_api_cache_file = Path("external_api_rates_cache.json")
        self.load_external_api_cache()

        # IBKR Data Manager for historical MIDPOINT forex rates (wired from main.py)
        self.ibkr_data_manager = None

        # Load fallback rates from config (user-configurable overrides)
        self._load_fallback_rates()

    def _load_fallback_rates(self):
        """
        Load fallback currency rates from config.py.
        These are used when IBKR and IBKR historical both fail.
        """
        try:
            from config import CURRENCY_RATE_FALLBACKS

            self.fallback_rates = CURRENCY_RATE_FALLBACKS
            self.logger.info(
                f"Loaded {len(self.fallback_rates)} currency rate overrides from config"
            )
        except ImportError:
            self.logger.warning(
                "CURRENCY_RATE_FALLBACKS not found in config, using built-in defaults"
            )
            # Built-in defaults (same as what was hardcoded before)
            self.fallback_rates = {
                "CAD": 0.72,
                "GBP": 1.27,
                "EUR": 1.08,
                "AUD": 0.66,
                "NZD": 0.60,
                "CHF": 1.12,
                "SGD": 0.74,
                "JPY": 0.00645,
                "HKD": 0.128,
                "HUF": 0.00263,
                "SEK": 0.095,
                "CZK": 0.0426,
                "NOK": 0.0926,
                "PLN": 0.25,
                "DKK": 0.145,
                "ILS": 0.27,
                "MXN": 0.059,
                "ZAR": 0.054,
                "TRY": 0.031,
                "RON": 0.217,
                "INR": 0.012,
                "SAR": 0.267,
            }

    def get_fallback_rate(self, currency: str) -> Optional[float]:
        """
        Get the fallback rate for a currency from config.

        Args:
            currency: Currency code (e.g., 'CAD', 'JPY')

        Returns:
            XXX_TO_USD rate (how many USD per 1 unit of XXX), or None if not configured
        """
        return self.fallback_rates.get(currency)

    def get_next_tickerId(self) -> int:
        """Thread-safe ticker ID generation"""
        with self.tickerId_lock:
            tickerId = self.current_tickerId
            self.current_tickerId += 1
            return tickerId

    def create_forex_contract(
        self, base_currency: str, quote_currency: str
    ) -> Contract:
        """
        Create forex contract using official IBKR API specification.

        Official IBKR API Format for Forex/Cash:
        - secType: 'CASH'
        - symbol: base currency (e.g., 'GBP')
        - currency: quote currency (e.g., 'USD')
        - exchange: 'IDEALPRO' (standard for forex)

        Example: GBP/USD
        contract.symbol = 'GBP'
        contract.currency = 'USD'

        Args:
            base_currency: Base currency code (e.g., 'GBP', 'USD')
            quote_currency: Quote currency code (e.g., 'USD', 'JPY')

        Returns:
            ibapi.contract.Contract configured for forex pair
        """
        contract = Contract()
        contract.symbol = base_currency
        contract.secType = "CASH"
        contract.currency = quote_currency
        contract.exchange = "IDEALPRO"

        self.logger.debug(f"Created forex contract: {base_currency}/{quote_currency}")
        return contract

    def fetch_forex_rate(
        self, base: str, quote: str, timeout: float = 10.0
    ) -> Optional[float]:
        """
        Fetch forex rate from IBKR using market data snapshot.

        Official IBKR API Method:
        reqMktData(tickerId, contract, genericTickList, snapshot, regulatorySnapshot, mktDataOptions)

        Args:
            base: Base currency (e.g., 'GBP')
            quote: Quote currency (e.g., 'USD')
            timeout: Maximum time to wait for response

        Returns:
            Exchange rate or None if failed
        """
        pair = f"{base}{quote}"

        # Check cache if recent
        if self._is_rate_fresh(pair):
            self.logger.debug(f"Using cached forex rate for {pair}: {self.rates[pair]}")
            return self.rates[pair]

        # Generate ticker ID
        tickerId = self.get_next_tickerId()

        # Setup event for this request
        self.rate_events[tickerId] = threading.Event()
        self.pending_forex_requests[tickerId] = pair
        self.tick_data[tickerId] = {}

        # Create forex contract
        contract = self.create_forex_contract(base, quote)

        # Request market data snapshot
        # snapshot=True means one-time data request
        # Official IBKR API: reqMktData(tickerId, contract, genericTickList, snapshot, regulatorySnapshot, mktDataOptions)
        self.logger.info(f"Requesting forex rate for {pair} (tickerId={tickerId})")
        try:
            self.ib_client.reqMktData(
                tickerId,  # First positional argument (not keyword)
                contract,  # Second positional argument
                "",  # genericTickList - empty string
                True,  # snapshot - one-time data request
                False,  # regulatorySnapshot
                [],  # mktDataOptions - empty list
            )
            # Add delay after forex rate request (2-3 seconds)
            delay = random.uniform(2.0, 3.0)
            self.logger.debug(
                f"Waiting {delay:.2f}s after forex rate request for {pair}"
            )
            time.sleep(delay)
        except Exception as e:
            self.logger.error(f"Error requesting forex rate for {pair}: {e}")
            self._cleanup_forex_request(tickerId)
            return None

        # Wait for tick data
        if self.rate_events[tickerId].wait(timeout=timeout):
            # Calculate rate from tick data (use last price if available, else mid-price)
            ticks = self.tick_data.get(tickerId, {})
            rate = None

            if "last" in ticks:
                rate = ticks["last"]
            elif "bid" in ticks and "ask" in ticks:
                rate = (ticks["bid"] + ticks["ask"]) / 2
            elif "close" in ticks:
                rate = ticks["close"]

            if rate and rate > 0:  # Validate rate is positive
                self.rates[pair] = rate
                self.last_update = time.time()
                self.logger.info(f"Fetched forex rate {pair}: {rate:.6f}")
                self.save_cache()
                self._cleanup_forex_request(tickerId)
                return rate
            else:
                self.logger.warning(f"Invalid forex rate received: {pair}={rate}")
                self._cleanup_forex_request(tickerId)
                return None
        else:
            self.logger.error(f"Timeout fetching forex rate for {pair}")
            self._cleanup_forex_request(tickerId)
            return None

    def _is_rate_fresh(self, pair: str) -> bool:
        """Check if cached rate is still fresh"""
        if pair not in self.rates or not self.last_update:
            return False
        rate = self.rates[pair]
        if rate <= 0:  # Reject negative or zero rates
            return False
        age = time.time() - self.last_update
        return age < self.update_interval

    def _cleanup_forex_request(self, tickerId: int):
        """Clean up request tracking"""
        self.rate_events.pop(tickerId, None)
        self.pending_forex_requests.pop(tickerId, None)
        self.tick_data.pop(tickerId, None)

    def handle_tick_price(self, tickerId: int, tickType: int, price: float):
        """
        Callback handler for IBKR tickPrice.

        This method is called by IBClient when price ticks are received.

        Official IBKR API Callback:
        tickPrice(int tickerId, int tickType, double price, TickAttrib attribs)

        Args:
            tickerId: Ticker ID
            tickType: Type of tick (BID=1, ASK=2, LAST=4, CLOSE=9)
            price: Price value
        """
        if tickerId not in self.pending_forex_requests:
            return

        pair = self.pending_forex_requests[tickerId]

        # Store tick data
        if tickerId not in self.tick_data:
            self.tick_data[tickerId] = {}

        if tickType == self.TICK_BID:
            if price > 0:
                self.tick_data[tickerId]["bid"] = price
                self.logger.debug(f"{pair} BID: {price}")
            else:
                self.logger.warning(f"Invalid tick price for {pair} BID: {price}")
        elif tickType == self.TICK_ASK:
            if price > 0:
                self.tick_data[tickerId]["ask"] = price
                self.logger.debug(f"{pair} ASK: {price}")
            else:
                self.logger.warning(f"Invalid tick price for {pair} ASK: {price}")
        elif tickType == self.TICK_LAST:
            if price > 0:
                self.tick_data[tickerId]["last"] = price
                self.logger.debug(f"{pair} LAST: {price}")
            else:
                self.logger.warning(f"Invalid tick price for {pair} LAST: {price}")
        elif tickType == self.TICK_CLOSE:
            if price > 0:
                self.tick_data[tickerId]["close"] = price
                self.logger.debug(f"{pair} CLOSE: {price}")
            else:
                self.logger.warning(f"Invalid tick price for {pair} CLOSE: {price}")

        # Signal that we have at least one price
        if "last" in self.tick_data[tickerId] or (
            "bid" in self.tick_data[tickerId] and "ask" in self.tick_data[tickerId]
        ):
            self.rate_events[tickerId].set()

    def handle_tick_snapshot_end(self, tickerId: int):
        """
        Callback handler for snapshot completion.

        Official IBKR API Callback:
        tickSnapshotEnd(int tickerId)

        Args:
            tickerId: Ticker ID
        """
        if tickerId in self.rate_events:
            # Set event even if we don't have all data
            self.rate_events[tickerId].set()

    def _fetch_forex_from_ibkr_historical(
        self, base: str, quote: str
    ) -> Optional[float]:
        """Fetch forex rate from IBKR historical MIDPOINT data."""
        if self.ibkr_data_manager is None:
            return None
        try:
            return self.ibkr_data_manager.fetch_forex_rate(base, quote)
        except Exception as e:
            self.logger.debug(
                f"IBKR historical forex rate failed for {base}/{quote}: {e}"
            )
            return None

    def _fetch_forex_from_frankfurter(self, currency: str) -> Optional[float]:
        """
        Fetch forex rate from Frankfurter.dev API (3rd fallback source).

        Frankfurter.dev provides free forex rates from the European Central Bank.
        No API key required. Updates daily around 16:00 CET.

        API: https://api.frankfurter.dev/v1/latest?base=USD&symbols=XXX

        Args:
            currency: Target currency code (e.g., 'CAD', 'EUR', 'JPY')

        Returns:
            XXX_TO_USD rate (how many USD per 1 unit of XXX), or None if failed
        """
        cache_key = f"FRANKFURTER_{currency}"

        # Check cache first
        if cache_key in self.external_api_cache:
            age = time.time() - self.external_api_timestamp.get(cache_key, 0)
            if age < self.external_api_duration:
                rate = self.external_api_cache[cache_key]
                self.logger.debug(
                    f"Using cached Frankfurter rate for {currency}: {rate:.6f}"
                )
                return rate

        try:
            # Fetch from Frankfurter.dev API with USD as base
            url = f"https://api.frankfurter.dev/v1/latest?base=USD&symbols={currency}"
            self.logger.info(f"Fetching {currency} rate from Frankfurter.dev...")

            response = requests.get(url, timeout=15)

            if response.status_code == 200:
                data = response.json()
                # Response: {"base":"USD","date":"2024-11-25","rates":{"CAD":1.4}}
                # This gives us USD/XXX rate (how many XXX per 1 USD)
                usd_to_xxx = data["rates"].get(currency)

                if usd_to_xxx and usd_to_xxx > 0:
                    # Convert to XXX_TO_USD (how many USD per 1 XXX)
                    xxx_to_usd = 1.0 / usd_to_xxx

                    # Cache the XXX_TO_USD rate
                    self.external_api_cache[cache_key] = xxx_to_usd
                    self.external_api_timestamp[cache_key] = time.time()
                    self.save_external_api_cache()

                    self.logger.info(
                        f"Fetched {currency} from Frankfurter: 1 {currency} = ${xxx_to_usd:.6f} USD"
                    )
                    return xxx_to_usd
                else:
                    self.logger.warning(
                        f"Frankfurter returned invalid rate for {currency}: {usd_to_xxx}"
                    )
            else:
                self.logger.warning(
                    f"Frankfurter API returned status {response.status_code}"
                )

        except Exception as e:
            self.logger.warning(f"Failed to fetch {currency} from Frankfurter: {e}")

            # Try stale cache
            if cache_key in self.external_api_cache:
                rate = self.external_api_cache[cache_key]
                age = time.time() - self.external_api_timestamp.get(cache_key, 0)
                self.logger.warning(
                    f"Using stale Frankfurter cache for {currency}: {rate:.6f} (age: {age / 3600:.1f} hours)"
                )
                return rate

        return None

    def convert_to_usd(self, amount: float, currency: str) -> Optional[float]:
        """
        Convert amount in any currency to USD.

        Args:
            amount: Amount in source currency
            currency: Source currency code (e.g., 'GBP', 'JPY', 'USD')

        Returns:
            Amount in USD or None if conversion failed
        """
        if currency == "USD":
            return amount

        # Fetch appropriate forex rate
        if currency == "GBP":
            # GBP to USD: multiply by GBP/USD rate
            rate = self.fetch_forex_rate("GBP", "USD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("GBP", "USD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_frankfurter("GBP")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} GBP to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            return self._fallback_convert_to_usd_multiply(amount, currency, 1.27)

        elif currency == "JPY":
            # JPY to USD: divide by USD/JPY rate
            rate = self.fetch_forex_rate("USD", "JPY")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "JPY")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} JPY to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            # Try Frankfurter (returns XXX_TO_USD directly)
            jpy_to_usd = self._fetch_forex_from_frankfurter("JPY")
            if jpy_to_usd and jpy_to_usd > 0:
                usd_amount = amount * jpy_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} JPY to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 155.0)

        elif currency == "EUR":
            # EUR to USD: multiply by EUR/USD rate
            rate = self.fetch_forex_rate("EUR", "USD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("EUR", "USD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_frankfurter("EUR")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} EUR to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            return self._fallback_convert_to_usd_multiply(amount, currency, 1.08)

        elif currency == "HKD":
            # HKD to USD: divide by USD/HKD rate
            rate = self.fetch_forex_rate("USD", "HKD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "HKD")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} HKD to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            # Try Frankfurter (returns XXX_TO_USD directly)
            hkd_to_usd = self._fetch_forex_from_frankfurter("HKD")
            if hkd_to_usd and hkd_to_usd > 0:
                usd_amount = amount * hkd_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} HKD to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 7.80)

        elif currency == "AUD":
            # AUD to USD: multiply by AUD/USD rate
            rate = self.fetch_forex_rate("AUD", "USD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("AUD", "USD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_frankfurter("AUD")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} AUD to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            return self._fallback_convert_to_usd_multiply(amount, currency, 0.66)

        elif currency == "SGD":
            # SGD to USD: IDEALPRO pair is USD.SGD (USD is base, SGD is quote)
            # Fetch USD/SGD rate and invert to get SGD->USD
            rate = self.fetch_forex_rate("USD", "SGD")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} SGD to ${usd_amount:.2f} USD (USD/SGD rate: {rate:.6f})"
                )
                return usd_amount
            # IBKR historical fallback: SGD/USD directly
            rate = self._fetch_forex_from_ibkr_historical("SGD", "USD")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} SGD to ${usd_amount:.2f} USD (IBKR historical rate: {rate:.6f})"
                )
                return usd_amount
            rate = self._fetch_forex_from_frankfurter("SGD")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} SGD to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd_multiply(amount, currency, 0.74)

        elif currency == "CAD":
            # CAD to USD: IDEALPRO pair is USD.CAD (USD is base, CAD is quote)
            # Fetch USD/CAD rate and invert to get CAD->USD
            rate = self.fetch_forex_rate("USD", "CAD")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} CAD to ${usd_amount:.2f} USD (USD/CAD rate: {rate:.6f})"
                )
                return usd_amount
            # IBKR historical fallback: CAD/USD directly
            rate = self._fetch_forex_from_ibkr_historical("CAD", "USD")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} CAD to ${usd_amount:.2f} USD (IBKR historical rate: {rate:.6f})"
                )
                return usd_amount
            rate = self._fetch_forex_from_frankfurter("CAD")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} CAD to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd_multiply(amount, currency, 0.72)

        elif currency == "CHF":
            # CHF to USD: IDEALPRO pair is USD.CHF (USD is base, CHF is quote)
            # Fetch USD/CHF rate and invert to get CHF->USD
            rate = self.fetch_forex_rate("USD", "CHF")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} CHF to ${usd_amount:.2f} USD (USD/CHF rate: {rate:.6f})"
                )
                return usd_amount
            # IBKR historical fallback: CHF/USD directly
            rate = self._fetch_forex_from_ibkr_historical("CHF", "USD")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} CHF to ${usd_amount:.2f} USD (IBKR historical rate: {rate:.6f})"
                )
                return usd_amount
            rate = self._fetch_forex_from_frankfurter("CHF")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} CHF to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd_multiply(amount, currency, 1.12)

        elif currency == "NZD":
            # NZD to USD: multiply by NZD/USD rate
            rate = self.fetch_forex_rate("NZD", "USD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("NZD", "USD")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_frankfurter("NZD")
            if rate and rate > 0:
                usd_amount = amount * rate
                self.logger.debug(
                    f"Converted {amount:.2f} NZD to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            return self._fallback_convert_to_usd_multiply(amount, currency, 0.60)

        elif currency == "INR":
            # INR to USD: NOT available on IBKR IDEALPRO, use external API
            # USD/INR is not supported by IBKR for forex trading
            return self._convert_inr_to_usd_external(amount)

        elif currency == "HUF":
            # HUF to USD: divide by USD/HUF rate
            rate = self.fetch_forex_rate("USD", "HUF")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "HUF")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} HUF to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            # Frankfurter returns XXX_TO_USD directly
            huf_to_usd = self._fetch_forex_from_frankfurter("HUF")
            if huf_to_usd and huf_to_usd > 0:
                usd_amount = amount * huf_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} HUF to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 380.0)

        elif currency == "SEK":
            # SEK to USD: divide by USD/SEK rate
            rate = self.fetch_forex_rate("USD", "SEK")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "SEK")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} SEK to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            sek_to_usd = self._fetch_forex_from_frankfurter("SEK")
            if sek_to_usd and sek_to_usd > 0:
                usd_amount = amount * sek_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} SEK to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 10.5)

        elif currency == "CZK":
            # CZK to USD: divide by USD/CZK rate
            rate = self.fetch_forex_rate("USD", "CZK")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "CZK")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} CZK to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            czk_to_usd = self._fetch_forex_from_frankfurter("CZK")
            if czk_to_usd and czk_to_usd > 0:
                usd_amount = amount * czk_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} CZK to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 23.5)

        elif currency == "NOK":
            # NOK to USD: divide by USD/NOK rate
            rate = self.fetch_forex_rate("USD", "NOK")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "NOK")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} NOK to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            nok_to_usd = self._fetch_forex_from_frankfurter("NOK")
            if nok_to_usd and nok_to_usd > 0:
                usd_amount = amount * nok_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} NOK to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 10.8)

        elif currency == "PLN":
            # PLN to USD: divide by USD/PLN rate
            rate = self.fetch_forex_rate("USD", "PLN")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "PLN")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} PLN to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            pln_to_usd = self._fetch_forex_from_frankfurter("PLN")
            if pln_to_usd and pln_to_usd > 0:
                usd_amount = amount * pln_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} PLN to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 4.0)

        elif currency == "DKK":
            # DKK to USD: divide by USD/DKK rate
            rate = self.fetch_forex_rate("USD", "DKK")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "DKK")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} DKK to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            dkk_to_usd = self._fetch_forex_from_frankfurter("DKK")
            if dkk_to_usd and dkk_to_usd > 0:
                usd_amount = amount * dkk_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} DKK to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 6.9)

        elif currency == "ILS":
            # ILS to USD: divide by USD/ILS rate
            rate = self.fetch_forex_rate("USD", "ILS")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "ILS")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} ILS to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            ils_to_usd = self._fetch_forex_from_frankfurter("ILS")
            if ils_to_usd and ils_to_usd > 0:
                usd_amount = amount * ils_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} ILS to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 3.7)

        elif currency == "MXN":
            # MXN to USD: divide by USD/MXN rate
            rate = self.fetch_forex_rate("USD", "MXN")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "MXN")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} MXN to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            mxn_to_usd = self._fetch_forex_from_frankfurter("MXN")
            if mxn_to_usd and mxn_to_usd > 0:
                usd_amount = amount * mxn_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} MXN to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 17.0)

        elif currency == "ZAR":
            # ZAR to USD: divide by USD/ZAR rate
            rate = self.fetch_forex_rate("USD", "ZAR")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "ZAR")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} ZAR to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            zar_to_usd = self._fetch_forex_from_frankfurter("ZAR")
            if zar_to_usd and zar_to_usd > 0:
                usd_amount = amount * zar_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} ZAR to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 18.5)

        elif currency == "TRY":
            # TRY to USD: divide by USD/TRY rate
            rate = self.fetch_forex_rate("USD", "TRY")
            if not rate or rate <= 0:
                rate = self._fetch_forex_from_ibkr_historical("USD", "TRY")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} TRY to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            try_to_usd = self._fetch_forex_from_frankfurter("TRY")
            if try_to_usd and try_to_usd > 0:
                usd_amount = amount * try_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} TRY to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 32.0)

        elif currency == "RON":
            # RON to USD: divide by USD/RON rate (NOT on IDEALPRO)
            rate = self._fetch_forex_from_ibkr_historical("USD", "RON")
            if rate and rate > 0:
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} RON to ${usd_amount:.2f} USD (rate: {rate:.6f})"
                )
                return usd_amount
            ron_to_usd = self._fetch_forex_from_frankfurter("RON")
            if ron_to_usd and ron_to_usd > 0:
                usd_amount = amount * ron_to_usd
                self.logger.debug(
                    f"Converted {amount:.2f} RON to ${usd_amount:.2f} USD (Frankfurter rate)"
                )
                return usd_amount
            return self._fallback_convert_to_usd(amount, currency, 4.6)

        elif currency == "SAR":
            # SAR to USD: Saudi Riyal is PEGGED to USD at 3.75
            # Not available on IDEALPRO, use fixed rate
            rate = 3.75  # Official peg rate
            usd_amount = amount / rate
            self.logger.debug(
                f"Converted {amount:.2f} SAR to ${usd_amount:.2f} USD (pegged rate: {rate})"
            )
            return usd_amount

        self.logger.error(f"Currency conversion not supported for {currency}")
        return None

    def _fallback_convert_to_usd(
        self, amount: float, currency: str, default_rate: float
    ) -> float:
        """
        Fallback conversion using config rate (or default) when all other methods fail.
        Uses DIVISION (for currencies quoted as USD/XXX, e.g., USD/JPY=155).

        Args:
            amount: Amount in source currency
            currency: Source currency code
            default_rate: Default fallback rate if not in config (units of currency per 1 USD)

        Returns:
            Amount in USD (never returns None)
        """
        # Try to get rate from config first
        config_rate = self.get_fallback_rate(currency)
        if config_rate is not None:
            # Config stores XXX_TO_USD (how many USD per 1 XXX)
            # So we multiply, not divide
            usd_amount = amount * config_rate
            self.logger.warning(
                f"Using config fallback rate for {currency}: {config_rate} (XXX_TO_USD)"
            )
        else:
            # Use default rate (USD/XXX format, so divide)
            usd_amount = amount / default_rate
            self.logger.warning(
                f"Using built-in fallback rate for {currency}: 1/{default_rate} (USD/{currency})"
            )

        self.logger.warning(
            f"Converted {amount:.2f} {currency} to ${usd_amount:.2f} USD (FALLBACK - may be inaccurate)"
        )
        return usd_amount

    def _fallback_convert_to_usd_multiply(
        self, amount: float, currency: str, default_rate: float
    ) -> float:
        """
        Fallback conversion using config rate (or default) when all other methods fail.
        Uses MULTIPLICATION (for currencies quoted as XXX/USD, e.g., CAD/USD=0.72).

        Args:
            amount: Amount in source currency
            currency: Source currency code
            default_rate: Default fallback rate if not in config (USD per 1 unit of currency)

        Returns:
            Amount in USD (never returns None)
        """
        # Try to get rate from config first (config always stores XXX_TO_USD)
        config_rate = self.get_fallback_rate(currency)
        if config_rate is not None:
            rate = config_rate
            self.logger.warning(
                f"Using config fallback rate for {currency}: {rate} (XXX_TO_USD)"
            )
        else:
            rate = default_rate
            self.logger.warning(
                f"Using built-in fallback rate for {currency}: {rate} (XXX_TO_USD)"
            )

        usd_amount = amount * rate
        self.logger.warning(
            f"Converted {amount:.2f} {currency} to ${usd_amount:.2f} USD (FALLBACK - may be inaccurate)"
        )
        return usd_amount

    def _convert_inr_to_usd_external(self, amount: float) -> Optional[float]:
        """
        Convert INR to USD using external API (Frankfurter.dev).

        IBKR does NOT support USD/INR on IDEALPRO exchange.
        Uses free external API with persistent caching, proxy support, and intelligent fallback.

        Args:
            amount: Amount in INR

        Returns:
            Amount in USD or hardcoded fallback rate (NEVER returns None to prevent INR=USD bug)
        """
        cache_key = "USDINR"

        # Check cache first (even if stale, prefer it to failed API calls)
        if cache_key in self.external_api_cache:
            age = time.time() - self.external_api_timestamp.get(cache_key, 0)
            if age < self.external_api_duration:
                rate = self.external_api_cache[cache_key]
                usd_amount = amount / rate
                self.logger.debug(
                    f"Converted {amount:.2f} INR to ${usd_amount:.2f} USD (cached rate: {rate:.6f})"
                )
                return usd_amount

        # Fetch from Frankfurter.dev API (free, no API key required)
        try:
            url = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR"

            # Requests will automatically use proxy from environment variables (http_proxy/https_proxy)
            # No need to hardcode proxy - just let requests library pick it up
            response = requests.get(
                url, timeout=15
            )  # 15s timeout for network latency via proxy

            if response.status_code == 200:
                data = response.json()
                rate = data["rates"]["INR"]  # e.g., 83.25 INR per USD

                # Cache the rate
                self.external_api_cache[cache_key] = rate
                self.external_api_timestamp[cache_key] = time.time()
                self.save_external_api_cache()  # Persist to disk

                usd_amount = amount / rate
                self.logger.info(
                    f"Converted {amount:.2f} INR to ${usd_amount:.2f} USD (external API rate: {rate:.6f})"
                )
                return usd_amount
            else:
                self.logger.error(
                    f"External API returned status {response.status_code}"
                )
                raise Exception(f"API returned {response.status_code}")

        except Exception as e:
            self.logger.error(f"Failed to fetch USD/INR rate from external API: {e}")

            # Fallback 1: Use stale cache if available (better than wrong USD value)
            if cache_key in self.external_api_cache:
                rate = self.external_api_cache[cache_key]
                usd_amount = amount / rate
                age = time.time() - self.external_api_timestamp.get(cache_key, 0)
                self.logger.warning(
                    f"Using stale cached USD/INR rate: {rate:.6f} (age: {age / 3600:.1f} hours)"
                )
                return usd_amount

            # Fallback 2: Use hardcoded default rate (better than treating INR as USD)
            DEFAULT_USDINR_RATE = 83.0  # Conservative approximation
            usd_amount = amount / DEFAULT_USDINR_RATE
            self.logger.error(
                f"Using hardcoded fallback USD/INR rate: {DEFAULT_USDINR_RATE:.2f}"
            )
            self.logger.error(
                f"CRITICAL: Converting {amount:.2f} INR to ${usd_amount:.2f} USD using fallback rate"
            )
            self.logger.error(
                f"Position sizing may be inaccurate! Recommend manual verification."
            )
            return usd_amount

    def load_cache(self):
        """Load forex rates from cache file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cache_data = json.load(f)
                # Filter out invalid rates
                all_rates = cache_data.get("rates", {})
                self.rates = {k: v for k, v in all_rates.items() if v > 0}
                self.last_update = cache_data.get("last_update")
                invalid_count = len(all_rates) - len(self.rates)
                if invalid_count > 0:
                    self.logger.warning(
                        f"Filtered out {invalid_count} invalid forex rates from cache"
                    )
                self.logger.info(
                    f"Loaded {len(self.rates)} valid forex rates from cache"
                )
            except Exception as e:
                self.logger.error(f"Error loading forex rates cache: {e}")
                self.rates = {}
                self.last_update = None

    def save_cache(self):
        """Save forex rates to cache file"""
        try:
            cache_data = {"rates": self.rates, "last_update": self.last_update}
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            self.logger.debug(f"Saved {len(self.rates)} forex rates to cache")
        except Exception as e:
            self.logger.error(f"Error saving forex rates cache: {e}")

    def load_external_api_cache(self):
        """Load external API rates (USD/INR, etc.) from persistent cache"""
        if self.external_api_cache_file.exists():
            try:
                with open(self.external_api_cache_file, "r") as f:
                    cache_data = json.load(f)
                self.external_api_cache = cache_data.get("rates", {})
                self.external_api_timestamp = cache_data.get("timestamps", {})
                if self.external_api_cache:
                    self.logger.info(
                        f"Loaded {len(self.external_api_cache)} external API rates from cache"
                    )
            except Exception as e:
                self.logger.error(f"Error loading external API cache: {e}")
                self.external_api_cache = {}
                self.external_api_timestamp = {}

    def save_external_api_cache(self):
        """Save external API rates to persistent cache"""
        try:
            cache_data = {
                "rates": self.external_api_cache,
                "timestamps": self.external_api_timestamp,
            }
            with open(self.external_api_cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            self.logger.debug(
                f"Saved {len(self.external_api_cache)} external API rates to cache"
            )
        except Exception as e:
            self.logger.error(f"Error saving external API cache: {e}")
