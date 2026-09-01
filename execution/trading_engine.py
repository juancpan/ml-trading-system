"""
Optimized trading engine with improved architecture and efficiency.
Consolidates functionality and reduces redundancy.
"""

import time
import zmq
import sys
import os
import datetime as dt
import pytz
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from holidays import NYSE
import pandas as pd
import numpy as np
import yfinance as yf
import pickle

from ibapi.contract import Contract
from ibapi.order import Order

# Import configurations
from config import (
    LOG_FILE,
    LOG_LEVEL,
    SYMBOLS,
    ASSET_SPECIFIC_CONFIGS,
    REPORT_FILE,
    TRADING_HOUR_EST,
    TRADING_MINUTE_EST,
    TIMEZONE,
)


class TradingState(Enum):
    """Trading system states."""

    INITIALIZING = "initializing"
    CONNECTED = "connected"
    TRADING = "trading"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class MarketData:
    """Container for market data."""

    symbol: str
    timestamp: dt.datetime
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last


@dataclass
class Position:
    """Container for position information."""

    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.quantity * self.current_price

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.unrealized_pnl + self.realized_pnl


class DataManager:
    """
    Optimized data manager with caching and efficient data handling.
    """

    def __init__(self, logger, cache_size: int = 100):
        self.logger = logger
        self.cache_size = cache_size

        # Historical data cache
        self.historical_cache: Dict[str, pd.DataFrame] = {}
        self.cache_timestamps: Dict[str, dt.datetime] = {}
        self.cache_ttl = dt.timedelta(hours=1)

        # Real-time data
        self.market_data: Dict[str, MarketData] = {}
        self.data_lock = threading.Lock()

    def fetch_historical_data(
        self, symbol: str, end_date: dt.date, lookback_days: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data with caching.
        """
        cache_key = f"{symbol}_{end_date}"

        # Check cache
        if cache_key in self.historical_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and (dt.datetime.now() - cache_time) < self.cache_ttl:
                self.logger.debug(f"Using cached data for {symbol}")
                return self.historical_cache[cache_key]

        # Fetch new data
        start_date = end_date - dt.timedelta(days=lookback_days)

        try:
            self.logger.info(f"Fetching historical data for {symbol}")
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

            if not df.empty:
                # Process and cache
                df = self._process_historical_data(df)
                self.historical_cache[cache_key] = df
                self.cache_timestamps[cache_key] = dt.datetime.now()

                # Manage cache size
                if len(self.historical_cache) > self.cache_size:
                    oldest_key = min(
                        self.cache_timestamps, key=self.cache_timestamps.get
                    )
                    del self.historical_cache[oldest_key]
                    del self.cache_timestamps[oldest_key]

                return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")

        return None

    def _process_historical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw historical data."""
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        # Ensure required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[col for col in required_cols if col in df.columns]]

        # Convert to numeric and handle NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()
        return df.sort_index()

    def update_market_data(self, symbol: str, data: MarketData):
        """Thread-safe update of market data."""
        with self.data_lock:
            self.market_data[symbol] = data

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Thread-safe retrieval of market data."""
        with self.data_lock:
            return self.market_data.get(symbol)

    def get_features_for_strategy(
        self, symbol: str, lags: int = 5
    ) -> Optional[np.ndarray]:
        """
        Generate features for ML strategy.
        """
        cache_key = f"{symbol}_{dt.date.today()}"

        if cache_key not in self.historical_cache:
            # Fetch data if not in cache
            self.fetch_historical_data(symbol, dt.date.today() + dt.timedelta(days=1))

        df = self.historical_cache.get(cache_key)
        if df is None or len(df) < lags + 1:
            self.logger.warning(f"Insufficient data for {symbol}")
            return None

        # Calculate returns and create lagged features
        df["returns"] = np.log(df["Close"] / df["Close"].shift(1))

        features = []
        for lag in range(1, lags + 1):
            df[f"lag_{lag}"] = df["returns"].shift(lag)
            features.append(f"lag_{lag}")

        # Get latest complete row of features
        latest_features = df[features].dropna().tail(1)

        if latest_features.empty:
            return None

        return latest_features.values.reshape(1, -1)


class StrategyExecutor:
    """
    Optimized strategy executor with model caching.
    """

    def __init__(self, data_manager: DataManager, logger, lags: int = 5):
        self.data_manager = data_manager
        self.logger = logger
        self.lags = lags
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load all strategy models."""
        for symbol in SYMBOLS:
            if symbol not in ASSET_SPECIFIC_CONFIGS:
                self.logger.warning(f"No config for {symbol}")
                continue

            model_path = ASSET_SPECIFIC_CONFIGS[symbol]["strategy_model_path"]

            try:
                if os.path.exists(model_path):
                    import warnings

                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with open(model_path, "rb") as f:
                            self.models[symbol] = pickle.load(f)
                        for w in caught:
                            self.logger.warning(
                                f"Pickle load warning for {symbol}: {w.message}"
                            )
                    self.logger.info(f"Loaded model for {symbol}")
                else:
                    # Create dummy model
                    self.models[symbol] = self._create_dummy_model()
                    self.logger.warning(f"Using dummy model for {symbol}")

            except Exception as e:
                self.logger.error(f"Error loading model for {symbol}: {e}")
                self.models[symbol] = self._create_dummy_model()

    def _create_dummy_model(self):
        """Create a dummy model for testing."""

        class DummyModel:
            def predict(self, features):
                if features is not None and features.size > 0:
                    return np.array([1 if features[0][0] > 0 else -1])
                return np.array([0])

        return DummyModel()

    def generate_signals(self, symbols: List[str]) -> Dict[str, int]:
        """
        Generate trading signals for multiple symbols.
        """
        signals = {}

        for symbol in symbols:
            if symbol not in self.models:
                signals[symbol] = 0
                continue

            features = self.data_manager.get_features_for_strategy(symbol, self.lags)

            if features is None:
                self.logger.warning(f"No features for {symbol}")
                signals[symbol] = 0
                continue

            try:
                model = self.models[symbol]
                signal = model.predict(features)[0]
                signals[symbol] = int(signal)
                self.logger.info(f"Signal for {symbol}: {signal}")

            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                signals[symbol] = 0

        return signals


class PortfolioManager:
    """
    Optimized portfolio manager with efficient rebalancing.
    """

    def __init__(self, logger, initial_capital: float = 100000.0):
        self.logger = logger
        self.positions: Dict[str, Position] = {}
        self.account_values = {
            "NetLiquidation": initial_capital,
            "TotalCashValue": initial_capital,
            "GrossPositionValue": 0.0,
            "UnrealizedPnL": 0.0,
            "RealizedPnL": 0.0,
        }
        self.lock = threading.Lock()

    def update_position(self, symbol: str, quantity: int, avg_cost: float):
        """Update position information."""
        with self.lock:
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol)

            pos = self.positions[symbol]
            pos.quantity = quantity
            pos.avg_cost = avg_cost

    def update_market_price(self, symbol: str, price: float):
        """Update market price for position."""
        with self.lock:
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

    def get_rebalance_trades(
        self, signals: Dict[str, int], data_manager: DataManager
    ) -> Dict[str, int]:
        """
        Calculate trades needed for rebalancing based on signals.
        """
        trades = {}

        with self.lock:
            total_value = self.account_values["NetLiquidation"]

            # Calculate target allocations based on Kelly criterion
            target_allocations = {}
            for symbol in signals:
                if symbol not in ASSET_SPECIFIC_CONFIGS:
                    continue

                config = ASSET_SPECIFIC_CONFIGS[symbol]
                kelly_leverage = config.get("kelly_leverage", 1.0)

                # Adjust allocation based on signal
                signal = signals[symbol]
                if signal != 0:
                    target_allocations[symbol] = kelly_leverage * signal / len(signals)
                else:
                    target_allocations[symbol] = 0

            # Normalize allocations
            total_allocation = sum(abs(v) for v in target_allocations.values())
            if total_allocation > 1.0:
                factor = 1.0 / total_allocation
                target_allocations = {
                    k: v * factor for k, v in target_allocations.items()
                }

            # Calculate required trades
            for symbol, target_alloc in target_allocations.items():
                target_value = total_value * target_alloc

                # Get current position
                current_pos = self.positions.get(symbol, Position(symbol=symbol))
                current_value = current_pos.market_value

                # Get current price
                market_data = data_manager.get_market_data(symbol)
                if market_data and market_data.last > 0:
                    price = market_data.last
                else:
                    # Fall back to historical close
                    hist_data = data_manager.fetch_historical_data(
                        symbol, dt.date.today() + dt.timedelta(days=1), lookback_days=5
                    )
                    if hist_data is not None and not hist_data.empty:
                        price = hist_data["Close"].iloc[-1]
                    else:
                        self.logger.warning(f"No price available for {symbol}")
                        continue

                # Calculate trade quantity
                value_diff = target_value - current_value
                if abs(value_diff) > total_value * 0.01:  # 1% threshold
                    quantity = int(value_diff / price)
                    if quantity != 0:
                        trades[symbol] = quantity

        return trades

    def update_account_values(self, values: Dict[str, float]):
        """Update account values."""
        with self.lock:
            self.account_values.update(values)


class OptimizedTradingEngine:
    """
    Main trading engine with improved architecture.
    """

    def __init__(self):
        self.state = TradingState.INITIALIZING
        self.logger = None
        self.zmq_context = None
        self.zmq_socket = None

        # Core components
        self.data_manager = None
        self.portfolio_manager = None
        self.strategy_executor = None
        self.ib_client = None

        # Trading configuration
        self.us_holidays = NYSE(
            years=range(dt.datetime.now().year - 1, dt.datetime.now().year + 2)
        )
        self.last_trading_day = None
        self.shutdown_event = threading.Event()

    def initialize(self):
        """Initialize all components."""
        try:
            # Setup ZMQ
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.bind("tcp://*:5556")

            # Setup logging
            from utils import setup_logger

            self.logger = setup_logger(
                LOG_FILE, level=LOG_LEVEL, zmq_pub_socket=self.zmq_socket
            )
            self.logger.info("Initializing trading engine...")

            # Initialize components
            self.data_manager = DataManager(self.logger)
            self.portfolio_manager = PortfolioManager(self.logger)
            self.strategy_executor = StrategyExecutor(self.data_manager, self.logger)

            # Initialize IB client
            from ib_client import IBClient

            self.ib_client = IBClient(
                self.data_manager,
                self.portfolio_manager,
                None,  # Order manager integrated into IB client
                self.logger,
            )

            self.state = TradingState.CONNECTED
            self.logger.info("Trading engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.state = TradingState.ERROR
            return False

    def connect_to_ib(self) -> bool:
        """Connect to Interactive Brokers."""
        if not self.ib_client.connect_and_run():
            self.logger.critical("Failed to connect to IBKR")
            return False

        # Request initial data
        if not self.ib_client.request_next_valid_id():
            self.logger.critical("Could not get valid order ID")
            return False

        self.ib_client.request_account_updates()
        time.sleep(3)  # Wait for initial updates

        # Subscribe to market data
        contracts = {}
        for symbol in SYMBOLS:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            contracts[symbol] = contract

        self.ib_client.request_market_data_for_symbols(SYMBOLS, contracts)
        time.sleep(2)  # Wait for initial market data

        return True

    def should_trade_today(self) -> bool:
        """Check if we should trade today."""
        current_time = dt.datetime.now(pytz.timezone(TIMEZONE))
        today = current_time.date()

        # Check if already processed today
        if today == self.last_trading_day:
            return False

        # Check time
        if current_time.hour < TRADING_HOUR_EST or (
            current_time.hour == TRADING_HOUR_EST
            and current_time.minute < TRADING_MINUTE_EST
        ):
            return False

        # Check weekend
        if today.weekday() in [5, 6]:
            self.logger.info(f"Skipping {today}: Weekend")
            return False

        # Check holiday
        if today in self.us_holidays:
            self.logger.info(f"Skipping {today}: Holiday")
            return False

        return True

    def execute_daily_trading(self):
        """Execute daily trading logic."""
        try:
            self.state = TradingState.TRADING
            today = dt.date.today()
            self.last_trading_day = today

            self.logger.info(f"Executing daily trading for {today}")

            # Get previous trading day
            target_date = today - dt.timedelta(days=1)
            while target_date.weekday() in [5, 6] or target_date in self.us_holidays:
                target_date -= dt.timedelta(days=1)

            # Fetch historical data for all symbols
            for symbol in SYMBOLS:
                self.data_manager.fetch_historical_data(
                    symbol, target_date + dt.timedelta(days=1)
                )

            # Generate signals
            signals = self.strategy_executor.generate_signals(SYMBOLS)
            self.logger.info(f"Generated signals: {signals}")

            # Calculate rebalancing trades
            trades = self.portfolio_manager.get_rebalance_trades(
                signals, self.data_manager
            )

            if trades:
                self.logger.info(f"Executing trades: {trades}")
                self.execute_trades(trades, signals)
            else:
                self.logger.info("No trades needed")

            self.state = TradingState.CONNECTED

        except Exception as e:
            self.logger.error(f"Error in daily trading: {e}")
            self.state = TradingState.ERROR

    def execute_trades(self, trades: Dict[str, int], signals: Dict[str, int]):
        """Execute the calculated trades."""
        for symbol, quantity in trades.items():
            # Check signal alignment
            signal = signals.get(symbol, 0)
            action = "BUY" if quantity > 0 else "SELL"

            # Validate trade against signal
            if (action == "BUY" and signal == -1) or (action == "SELL" and signal == 1):
                self.logger.warning(f"Skipping {symbol}: Signal conflict")
                continue

            # Create and place order
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"

            order = Order()
            order.action = action
            order.totalQuantity = abs(quantity)
            order.orderType = "MKT"
            order.tif = "DAY"

            order_id = self.ib_client.nextValidOrderId
            self.ib_client.nextValidOrderId += 1

            self.ib_client.place_order(contract, order, order_id)
            self.logger.info(f"Placed {action} order for {abs(quantity)} {symbol}")

            time.sleep(0.5)  # Rate limiting

    def run(self):
        """Main trading loop."""
        if not self.initialize():
            return

        if not self.connect_to_ib():
            self.shutdown()
            return

        self.logger.info("Starting main trading loop")

        try:
            while not self.shutdown_event.is_set() and self.ib_client.isConnected():
                if self.should_trade_today():
                    self.execute_daily_trading()

                # Generate periodic reports
                if hasattr(self, "last_report_time"):
                    if time.time() - self.last_report_time > 300:  # 5 minutes
                        self.generate_report()
                        self.last_report_time = time.time()
                else:
                    self.last_report_time = time.time()

                # Sleep for check interval
                time.sleep(60)

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested")
        except Exception as e:
            self.logger.critical(f"Fatal error: {e}")
        finally:
            self.shutdown()

    def generate_report(self):
        """Generate trading report."""
        from utils import generate_report

        generate_report(
            REPORT_FILE,
            self.portfolio_manager.positions,
            {},  # Open orders
            self.portfolio_manager.account_values,
        )

    def shutdown(self):
        """Shutdown the trading engine."""
        self.logger.info("Shutting down trading engine...")
        self.state = TradingState.SHUTTING_DOWN

        # Disconnect from IB
        if self.ib_client:
            self.ib_client.disconnect_ib()

        # Close ZMQ
        if self.zmq_socket:
            self.zmq_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        self.logger.info("Shutdown complete")


def main():
    """Main entry point."""
    engine = OptimizedTradingEngine()
    engine.run()


if __name__ == "__main__":
    main()
