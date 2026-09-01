"""
Contract Details Manager for IBKR API

Fetches and caches contract details from IBKR, including:
- priceMagnifier: Converts price units (e.g., GBX to GBP)
- minTick: Minimum price increment
- longName: Full company name

Follows official IBKR API specification:
https://interactivebrokers.github.io/tws-api/contract_details.html
"""

import threading
import time
import json
from pathlib import Path
from typing import Dict, Optional
from ibapi.contract import Contract


class ContractDetailsManager:
    """
    Manages contract details fetching and caching from IBKR API.

    Official IBKR API Methods Used:
    - reqContractDetails(reqId, contract)
    - contractDetails(reqId, contractDetails) - callback
    - contractDetailsEnd(reqId) - callback
    """

    def __init__(self, ib_client, logger):
        """
        Initialize contract details manager.

        Args:
            ib_client: IBClient instance with connection to IB Gateway
            logger: Logger instance
        """
        self.ib_client = ib_client
        self.logger = logger

        # Cache for contract details {yfinance_symbol: details_dict}
        self.details_cache = {}

        # Threading for synchronous API calls
        self.details_events = {}  # {reqId: threading.Event()}
        self.pending_requests = {}  # {reqId: yfinance_symbol}

        # Request ID management (start from 5000 to avoid conflicts)
        self.current_reqId = 5000
        self.reqId_lock = threading.Lock()

        # Persistence
        self.cache_file = Path('contract_details_cache.json')
        self.load_cache()

    def get_next_reqId(self) -> int:
        """Thread-safe request ID generation"""
        with self.reqId_lock:
            reqId = self.current_reqId
            self.current_reqId += 1
            return reqId

    def fetch_contract_details(self, yfinance_symbol: str, contract: Contract, timeout: float = 10.0) -> Dict:
        """
        Fetch contract details from IBKR API.

        Official IBKR API Method:
        reqContractDetails(int reqId, Contract contract)

        Args:
            yfinance_symbol: Symbol in yfinance format (for caching)
            contract: ibapi.contract.Contract object
            timeout: Maximum time to wait for response

        Returns:
            Dictionary with contract details:
            {
                'priceMagnifier': int (100 for LSE stocks, 1 for others),
                'minTick': float,
                'longName': str,
                'currency': str,
                'exchange': str
            }
        """
        # Check cache first
        if yfinance_symbol in self.details_cache:
            self.logger.debug(f"Using cached contract details for {yfinance_symbol}")
            return self.details_cache[yfinance_symbol]

        # Generate request ID
        reqId = self.get_next_reqId()

        # Setup event for this request
        self.details_events[reqId] = threading.Event()
        self.pending_requests[reqId] = yfinance_symbol

        # Request contract details from IBKR
        self.logger.info(f"Requesting contract details for {yfinance_symbol} (reqId={reqId})")
        try:
            self.ib_client.reqContractDetails(reqId, contract)
        except Exception as e:
            self.logger.error(f"Error requesting contract details for {yfinance_symbol}: {e}")
            self._cleanup_request(reqId)
            return {}

        # Wait for callback
        if self.details_events[reqId].wait(timeout=timeout):
            details = self.details_cache.get(yfinance_symbol, {})
            self._cleanup_request(reqId)

            if details:
                self.logger.info(f"Received contract details for {yfinance_symbol}: "
                               f"priceMagnifier={details.get('priceMagnifier', 1)}")
                self.save_cache()
                return details
            else:
                self.logger.warning(f"No details received for {yfinance_symbol}")
                return {}
        else:
            self.logger.error(f"Timeout fetching contract details for {yfinance_symbol}")
            self._cleanup_request(reqId)
            return {}

    def _cleanup_request(self, reqId: int):
        """Clean up request tracking"""
        self.details_events.pop(reqId, None)
        self.pending_requests.pop(reqId, None)

    def handle_contract_details(self, reqId: int, contractDetails):
        """
        Callback handler for IBKR contractDetails.

        This method is called by IBClient when contract details are received.

        Official IBKR API Callback:
        contractDetails(int reqId, ContractDetails contractDetails)

        Args:
            reqId: Request ID
            contractDetails: ContractDetails object from IBKR API
        """
        yfinance_symbol = self.pending_requests.get(reqId)

        if not yfinance_symbol:
            self.logger.warning(f"Received contract details for unknown reqId {reqId}")
            return

        # Extract relevant fields from ContractDetails
        # According to official IBKR API:
        # https://interactivebrokers.github.io/tws-api/classIBApi_1_1ContractDetails.html
        details = {
            'priceMagnifier': contractDetails.priceMagnifier,  # int
            'minTick': contractDetails.minTick,  # double
            'longName': contractDetails.longName,  # string
            'currency': contractDetails.contract.currency,  # string
            'exchange': contractDetails.contract.exchange,  # string
            'conId': contractDetails.contract.conId,  # int
            'marketName': contractDetails.marketName,  # string
            'timestamp': time.time()
        }

        # Store in cache
        self.details_cache[yfinance_symbol] = details

        self.logger.debug(
            f"Stored contract details for {yfinance_symbol}: "
            f"priceMagnifier={details['priceMagnifier']}, "
            f"minTick={details['minTick']}, "
            f"currency={details['currency']}, "
            f"exchange={details['exchange']}"
        )

    def handle_contract_details_end(self, reqId: int):
        """
        Callback handler for IBKR contractDetailsEnd.

        Official IBKR API Callback:
        contractDetailsEnd(int reqId)

        Args:
            reqId: Request ID
        """
        if reqId in self.details_events:
            self.details_events[reqId].set()
            self.logger.debug(f"Contract details retrieval complete for reqId {reqId}")

    def get_details(self, yfinance_symbol: str) -> Dict:
        """
        Get cached contract details for a symbol.

        Args:
            yfinance_symbol: Symbol in yfinance format

        Returns:
            Dictionary with contract details or empty dict if not cached
        """
        return self.details_cache.get(yfinance_symbol, {})

    def get_price_magnifier(self, yfinance_symbol: str) -> int:
        """
        Get price magnifier for a symbol.

        Returns 1 if not found (no conversion needed).

        Args:
            yfinance_symbol: Symbol in yfinance format

        Returns:
            Price magnifier (100 for LSE stocks in pence, 1 for others)
        """
        details = self.get_details(yfinance_symbol)
        return details.get('priceMagnifier', 1)

    def load_cache(self):
        """Load contract details from cache file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.details_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.details_cache)} contract details from cache")
            except Exception as e:
                self.logger.error(f"Error loading contract details cache: {e}")
                self.details_cache = {}

    def save_cache(self):
        """Save contract details to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.details_cache, f, indent=2)
            self.logger.debug(f"Saved {len(self.details_cache)} contract details to cache")
        except Exception as e:
            self.logger.error(f"Error saving contract details cache: {e}")
