"""Intelligent LIMIT order submit/cancel/reprice execution engine."""

import time
from ibapi.order import Order


class LimitOrderEngine:
    """Executes LIMIT orders with live IBKR quotes and retry logic."""

    TERMINAL_STATUSES = {"Filled", "Cancelled", "ApiCancelled", "Rejected", "Inactive"}

    def __init__(
        self,
        ib_client,
        order_manager,
        market_data_manager,
        exchange_manager,
        contract_details_mgr,
        logger,
    ):
        self.ib_client = ib_client
        self.order_manager = order_manager
        self.market_data_manager = market_data_manager
        self.exchange_manager = exchange_manager
        self.contract_details_mgr = contract_details_mgr
        self.logger = logger

    def _select_base_price(self, action: str, quote: dict):
        bid = quote.get("bid")
        ask = quote.get("ask")
        last = quote.get("last")
        close = quote.get("close")
        fallback = last if last and last > 0 else close

        if action == "BUY":
            return ask if ask and ask > 0 else fallback
        return bid if bid and bid > 0 else fallback

    def _calculate_limit_price_native(
        self,
        symbol: str,
        action: str,
        quote: dict,
        strategy: str,
        abs_offset,
        pct_offset,
    ):
        bid = quote.get("bid")
        ask = quote.get("ask")

        if strategy == "MIDPOINT" and bid and ask and bid > 0 and ask > 0:
            limit_price_native = (bid + ask) / 2
        else:
            base_price = self._select_base_price(action, quote)
            if base_price is None or base_price <= 0:
                return None, None

            if strategy == "OFFSET_FROM_NBBO":
                use_pct_offset = pct_offset is not None and pct_offset != 0
                use_abs_offset = abs_offset is not None and abs_offset != 0

                if action == "BUY":
                    if use_pct_offset:
                        limit_price_native = base_price * (1 + pct_offset)
                    elif use_abs_offset:
                        limit_price_native = base_price + abs_offset
                    else:
                        limit_price_native = base_price
                else:
                    if use_pct_offset:
                        limit_price_native = base_price * (1 - pct_offset)
                    elif use_abs_offset:
                        limit_price_native = base_price - abs_offset
                    else:
                        limit_price_native = base_price
            else:
                # CROSS_SPREAD default: cross to opposite side for immediate fills.
                limit_price_native = base_price

        if limit_price_native is None or limit_price_native <= 0:
            return None, None

        tick_size = self.exchange_manager.get_tick_size(symbol, limit_price_native)
        limit_price_rounded = round(limit_price_native / tick_size) * tick_size
        return limit_price_rounded, tick_size

    def _wait_for_terminal(self, order_id: int, timeout_seconds: int):
        """Wait until order reaches terminal state or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if self.order_manager.is_order_terminal(order_id):
                return True
            time.sleep(1.0)
        return False

    def _wait_for_cancellation(self, order_id: int, timeout_seconds: int = 15):
        """Wait until order leaves open bucket after cancellation request."""
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if not self.order_manager.is_order_open(order_id):
                return True
            time.sleep(0.5)
        return False

    def submit_with_retry(
        self,
        symbol: str,
        contract,
        quantity: float,
        action: str,
        *,
        max_retries: int,
        fill_timeout_seconds: int,
        quote_timeout_seconds: int,
        price_strategy: str,
        abs_offset,
        pct_offset,
    ):
        """Submit LIMIT order with cancel-reprice-retry behavior."""
        remaining_quantity = float(quantity)
        attempts_total = max_retries + 1

        for attempt in range(1, attempts_total + 1):
            if remaining_quantity <= 0:
                return True, None

            quote = self.market_data_manager.request_snapshot(
                symbol,
                contract,
                timeout=quote_timeout_seconds,
            )
            if not quote:
                return False, f"No live IBKR quote for {symbol}"

            limit_price_native, tick_size = self._calculate_limit_price_native(
                symbol,
                action,
                quote,
                price_strategy,
                abs_offset,
                pct_offset,
            )
            if limit_price_native is None:
                return False, f"Unable to calculate valid limit price for {symbol}"

            price_magnifier = self.contract_details_mgr.get_price_magnifier(symbol)
            limit_price_for_order = limit_price_native * price_magnifier

            order = Order()
            order.action = action
            order.orderType = "LMT"
            order.totalQuantity = remaining_quantity
            order.lmtPrice = limit_price_for_order
            order.tif = "DAY"
            order.transmit = True

            order_id = self.ib_client.allocate_order_id()
            self.logger.info(
                f"{symbol} attempt {attempt}/{attempts_total}: "
                f"{action} {remaining_quantity:.0f} @ {limit_price_native:.4f} "
                f"(tick={tick_size}, magnifier={price_magnifier})"
            )
            self.ib_client.place_order(contract, order, order_id)

            if self._wait_for_terminal(order_id, fill_timeout_seconds):
                info = self.order_manager.get_order_info(order_id) or {}
                status = info.get("status")
                if status == "Filled":
                    return True, None

                remaining = info.get("remainingQuantity", 0)
                try:
                    remaining_quantity = float(remaining)
                except (TypeError, ValueError):
                    remaining_quantity = 0

                if status in {"Rejected", "Inactive"}:
                    return (
                        False,
                        f"Order {order_id} rejected/inactive for {symbol}: {status}",
                    )

                if status in {"Cancelled", "ApiCancelled"}:
                    if remaining_quantity <= 0:
                        return True, None
                    continue

                if status in self.TERMINAL_STATUSES:
                    return False, f"Terminal status {status} for order {order_id}"

            self.logger.warning(
                f"{symbol} order {order_id} not filled within {fill_timeout_seconds}s, cancelling and repricing"
            )
            self.ib_client.cancel_order(order_id)

            if not self._wait_for_cancellation(order_id):
                self.logger.warning(
                    f"Timed out waiting for cancellation of order {order_id}"
                )

            info = self.order_manager.get_order_info(order_id) or {}
            remaining = info.get("remainingQuantity", remaining_quantity)
            try:
                remaining_quantity = float(remaining)
            except (TypeError, ValueError):
                remaining_quantity = 0

        if remaining_quantity > 0:
            return (
                False,
                f"Unfilled remainder {remaining_quantity:.0f} shares after retries",
            )
        return True, None
