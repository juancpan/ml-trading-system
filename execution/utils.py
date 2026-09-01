# utils.py

import logging
from logging.handlers import TimedRotatingFileHandler
import datetime as dt
import zmq
import pytz  # New import


def setup_logger(log_file, level=logging.INFO, zmq_pub_socket=None):
    """
    Sets up a logger that writes to a file and optionally publishes to ZeroMQ.
    """
    logger = logging.getLogger("AlgoTradingLogger")
    logger.setLevel(level)
    logger.propagate = False

    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=7
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    if zmq_pub_socket:

        class ZeroMQHandler(logging.Handler):
            def __init__(self, socket):
                super().__init__()
                self.socket = socket
                self.socket_closed = False

            def emit(self, record):
                try:
                    # Check if socket is still valid before sending
                    if not self.socket_closed and self.socket:
                        msg = self.format(record)
                        self.socket.send_string(msg, zmq.NOBLOCK)
                except zmq.Again:
                    # Socket is busy, skip this message
                    pass
                except (zmq.ZMQError, AttributeError):
                    # Socket is closed or invalid, mark it and stop trying
                    self.socket_closed = True
                except Exception:
                    # Log other errors silently
                    pass

            def close(self):
                """Mark socket as closed to prevent further send attempts."""
                self.socket_closed = True
                super().close()

        zmq_handler = ZeroMQHandler(zmq_pub_socket)
        zmq_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(zmq_handler)

    return logger


def generate_report(file_path, current_portfolio, open_orders, account_values):
    """Generates a detailed trading report."""
    with open(file_path, "a") as f:
        f.write(f"\n--- Trading Report - {dt.datetime.now()} ---\n")
        f.write("Account Summary:\n")
        for key, data in account_values.items():
            f.write(f"  {key}: {data.get('value', 'N/A')} {data.get('currency', '')}\n")

        f.write("\nCurrent Portfolio:\n")
        if current_portfolio:
            for symbol, pos_data in current_portfolio.items():
                # Handle positions data - convert Decimal to float to avoid type errors
                position = float(pos_data.get("position", 0))
                avg_cost = float(pos_data.get("averageCost", 0))

                # Use IBKR-sourced market price from portfolio data (already corrected for priceMagnifier)
                current_price = None
                if "marketPrice" in pos_data:
                    current_price = float(pos_data.get("marketPrice", 0))
                    if current_price <= 0:
                        current_price = None

                # Calculate real-time unrealized PnL if we have current price
                if current_price and position != 0:
                    market_value = position * current_price
                    total_invested = position * avg_cost
                    unrealized_pnl = market_value - total_invested
                else:
                    # Fall back to stored unrealized PnL from IBKR
                    unrealized_pnl = float(pos_data.get("unrealizedPNL", 0))
                    current_price = (
                        float(pos_data.get("marketPrice", 0))
                        if "marketPrice" in pos_data
                        else None
                    )

                # Calculate PnL percentage
                pnl_percentage = 0
                if position > 0 and avg_cost > 0:
                    total_invested = position * avg_cost
                    pnl_percentage = (unrealized_pnl / total_invested) * 100

                # Build output string with available fields
                output = f"  {symbol}: Position={position}, AvgCost={avg_cost:.2f}"

                # Add current price if available
                if current_price:
                    output += f", CurrentPrice={current_price:.2f}"

                # Add market value if we can calculate it
                if current_price and position != 0:
                    output += f", MarketValue={position * current_price:.2f}"
                elif "marketValue" in pos_data:
                    output += f", MarketValue={pos_data['marketValue']:.2f}"

                output += f", UnrealizedPNL={unrealized_pnl:.2f}"
                output += f", PNL%={pnl_percentage:.2f}%\n"
                f.write(output)
        else:
            f.write("  No open positions.\n")

        f.write("\nOpen Orders:\n")
        if open_orders:
            for order_id, order_info in open_orders.items():
                f.write(
                    f"  OrderID {order_id}: Symbol={order_info.get('contract', {}).symbol},"
                    f" Action={order_info.get('action')},"
                    f" TotalQty={order_info.get('totalQuantity')},"
                    f" FilledQty={order_info.get('filledQuantity')},"
                    f" Status={order_info.get('status')}\n"
                )
        else:
            f.write("  No open orders.\n")
        f.write("------------------------------------------\n\n")


def setup_zmq_publisher(port=5555):
    """Create a ZeroMQ PUB socket bound to the given port.

    The publisher is monitoring-only. Two failure modes used to be fatal:
      1. A still-alive overlapping region session (e.g. EUROPE running until
         ~23:45) held the port when US/CANADA fired at 23:30/23:35, so bind()
         raised ZMQError "Address already in use" and main.py exited 1 WITHOUT
         trading (see logs/cron_US_2026061[02]_233000.log).
      2. TIME_WAIT lingering after an unclean shutdown.

    Mitigations:
      - Per-region ports (caller passes a distinct port per region) so two
        legitimately-overlapping regions never collide.
      - LINGER=0 so the socket releases the port immediately on close().
      - bind() is wrapped: if the port is genuinely occupied we raise a clear
        error to the caller, which decides whether to continue without the
        publisher (trading must never be blocked by a monitoring socket).
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.bind(f"tcp://0.0.0.0:{port}")
    except zmq.ZMQError:
        # Clean up the half-built socket/context so we don't leak an fd, then
        # re-raise for the caller to handle (e.g. continue without ZMQ).
        try:
            socket.close(linger=0)
            context.term()
        except Exception:
            pass
        raise
    return context, socket


# Helper to convert local time to target timezone time
def get_current_time_in_timezone(target_tz):
    return dt.datetime.now(dt.timezone.utc).astimezone(target_tz)
