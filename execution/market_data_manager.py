"""IBKR stock quote manager for synchronous bid/ask snapshots."""

import random
import threading
import time


class MarketDataManager:
    """Fetches live stock bid/ask data from IBKR reqMktData snapshots."""

    TICK_BID = 1
    TICK_ASK = 2
    TICK_LAST = 4
    TICK_CLOSE = 9

    MARKET_DATA_PERMISSION_ERRORS = {354}

    def __init__(self, ib_client, logger):
        self.ib_client = ib_client
        self.logger = logger
        self.current_ticker_id = 7000
        self.ticker_id_lock = threading.Lock()
        self.quote_events = {}
        self.quote_errors = {}
        self.pending_quote_requests = {}
        self.tick_data = {}

    def _next_ticker_id(self) -> int:
        with self.ticker_id_lock:
            ticker_id = self.current_ticker_id
            self.current_ticker_id += 1
        return ticker_id

    def _cleanup_quote_request(self, ticker_id: int):
        self.quote_events.pop(ticker_id, None)
        self.quote_errors.pop(ticker_id, None)
        self.pending_quote_requests.pop(ticker_id, None)
        self.tick_data.pop(ticker_id, None)

    def request_snapshot(self, symbol: str, contract, timeout: float = 10.0):
        """Request a one-shot IBKR market data snapshot for a symbol."""
        ticker_id = self._next_ticker_id()
        self.quote_events[ticker_id] = threading.Event()
        self.pending_quote_requests[ticker_id] = symbol
        self.tick_data[ticker_id] = {}

        self.logger.debug(
            f"Requesting IBKR quote snapshot for {symbol} (tickerId={ticker_id})"
        )

        try:
            self.ib_client.reqMktData(ticker_id, contract, "", True, False, [])
            time.sleep(random.uniform(0.2, 0.4))
        except Exception as e:
            self.logger.error(f"Failed reqMktData snapshot for {symbol}: {e}")
            self._cleanup_quote_request(ticker_id)
            return None

        event = self.quote_events[ticker_id]
        if not event.wait(timeout=timeout):
            self.logger.warning(
                f"IBKR quote timeout for {symbol} (tickerId={ticker_id})"
            )
            self._cleanup_quote_request(ticker_id)
            return None

        error_code = self.quote_errors.get(ticker_id)
        if error_code in self.MARKET_DATA_PERMISSION_ERRORS:
            self.logger.warning(
                f"No IBKR market data permission for {symbol} (error {error_code})"
            )
            self._cleanup_quote_request(ticker_id)
            return None

        data = self.tick_data.get(ticker_id, {})
        bid = data.get("bid")
        ask = data.get("ask")
        last = data.get("last")
        close = data.get("close")

        self._cleanup_quote_request(ticker_id)

        if bid is None and ask is None and last is None and close is None:
            self.logger.warning(f"IBKR snapshot returned no quote data for {symbol}")
            return None

        return {
            "bid": bid,
            "ask": ask,
            "last": last,
            "close": close,
        }

    def handle_tick_price(self, ticker_id: int, tick_type: int, price: float):
        """Handle asynchronous tickPrice callback updates."""
        if ticker_id not in self.pending_quote_requests:
            return

        if price is None or price <= 0:
            return

        if tick_type == self.TICK_BID:
            self.tick_data[ticker_id]["bid"] = price
        elif tick_type == self.TICK_ASK:
            self.tick_data[ticker_id]["ask"] = price
        elif tick_type == self.TICK_LAST:
            self.tick_data[ticker_id]["last"] = price
        elif tick_type == self.TICK_CLOSE:
            self.tick_data[ticker_id]["close"] = price

        data = self.tick_data.get(ticker_id, {})
        if data.get("last") is not None or (
            data.get("bid") is not None and data.get("ask") is not None
        ):
            event = self.quote_events.get(ticker_id)
            if event:
                event.set()

    def handle_tick_snapshot_end(self, ticker_id: int):
        """Unblock snapshot waiter when IBKR signals snapshot completion."""
        event = self.quote_events.get(ticker_id)
        if event:
            event.set()

    def handle_error(self, req_id: int, error_code: int, error_message: str):
        """Handle API errors tied to market data request IDs."""
        if req_id not in self.pending_quote_requests:
            return

        self.quote_errors[req_id] = error_code
        symbol = self.pending_quote_requests.get(req_id, "UNKNOWN")
        self.logger.warning(
            f"IBKR quote error for {symbol} (reqId={req_id}, code={error_code}): {error_message}"
        )

        event = self.quote_events.get(req_id)
        if event:
            event.set()
