"""
Portfolio Manager — uses IBKR as primary data source for prices and positions.
Fallback chain: IBKR historical → parquet store → IBKR snapshot → yfinance (last resort).
"""

import pickle
import json
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from ibapi.contract import Contract
from config import (
    SYMBOLS,
    TARGET_ALLOCATION,
    ASSET_SPECIFIC_CONFIGS,
    REBALANCE_THRESHOLD_PERCENT,
    MIN_TRADE_SHARES,
    BLACKLISTED_SYMBOLS,
)


class PortfolioManager:
    """
    Manages portfolio with yfinance price data and IBKR position data.
    """

    def __init__(
        self,
        logger,
        exchange_manager=None,
        contract_details_mgr=None,
        currency_converter=None,
    ):
        self.logger = logger

        # IBKR data - positions and account values
        self.current_positions = {}  # {symbol: {'position': float, 'averageCost': float, ...}}
        self.account_values = {}  # {key: {'value': str, 'currency': str, ...}}

        # Position loading control flag
        self.initial_position_load_mode = False  # Ignore position=0 during initial load

        # State file paths for oversight system
        self.state_dir = Path(__file__).parent
        self.account_state_file = self.state_dir / "account_values.pkl"
        self.positions_file = self.state_dir / "positions.pkl"
        self.json_state_file = self.state_dir / "account_state.json"

        # Configuration
        self.target_allocation = TARGET_ALLOCATION
        self.asset_configs = ASSET_SPECIFIC_CONFIGS

        # Dual price caches for multi-currency support
        self.latest_prices_usd = {}  # USD prices (for position sizing calculations)
        self.latest_prices_native = {}  # Native currency prices (for order execution)
        self.price_currencies = {}  # Track currency per symbol
        self.latest_prices = self.latest_prices_usd  # Backward compatibility alias
        self.last_price_update = None

        # Multi-currency support managers
        self.exchange_manager = exchange_manager
        self.contract_details_mgr = contract_details_mgr

        # Primary IBKR data source (set via main.py after IBKR connection)
        self.ibkr_data_manager = None

        # Fallback price source managers (wired after initialization)
        self.data_manager = None  # DataManager with parquet cache (set via main.py)
        self.market_data_manager = None  # IBKR MarketDataManager (set via main.py)
        self.currency_converter = currency_converter

        # Config-authoritative reverse map: bare IBKR symbol -> canonical config
        # key. IBKR returns bare symbols (e.g. "XYZ") with an empty/SMART
        # exchange on some position callbacks. Without this, update_position()
        # fell back to currency inference (EUR -> "SBF"/Paris), mislabeling
        # Milan tickers as ".PA" and storing the SAME instrument under two keys
        # (XYZ.MI AND XYZ.PA). This map makes the config key authoritative.
        # See MEMORY.md "European position stored under wrong exchange key (.PA)".
        self._bare_to_config = self._build_bare_symbol_map()

    def _build_bare_symbol_map(self):
        """Build {bare_symbol: config_key} from configured tickers.

        The bare symbol is everything before the first '.' (e.g. 'XYZ.MI' ->
        'XYZ'). On collision (two configured tickers share a bare symbol),
        the entry is set to None so resolution falls back to exchange/currency
        disambiguation rather than guessing wrong.
        """
        mapping: dict[str, str | None] = {}
        collisions: set[str] = set()
        for config_key in SYMBOLS:
            bare = config_key.split(".")[0]
            if not bare or bare == config_key:
                # US-style tickers (no suffix) map to themselves; harmless.
                mapping.setdefault(bare, config_key)
                continue
            if bare in mapping and mapping[bare] != config_key:
                collisions.add(bare)
                mapping[bare] = None  # ambiguous -> force fallback
            elif bare not in collisions:
                mapping[bare] = config_key
        if collisions:
            self.logger.warning(
                "Bare-symbol collisions (will use exchange/currency fallback): %s",
                sorted(collisions),
            )
        return mapping

    def update_account_value(self, key, val, currency, accountName):
        """Updates account values from IBKR."""
        try:
            val_parsed = float(val)
        except ValueError:
            val_parsed = val
        self.account_values[key] = {
            "value": val_parsed,
            "currency": currency,
            "accountName": accountName,
        }

        # Save state for oversight system
        self._save_state_for_oversight()

    def update_position(
        self,
        contract: Contract,
        position: float,
        marketPrice: float,
        marketValue: float,
        averageCost: float,
        unrealizedPNL: float,
        realizedPNL: float,
        accountName: str,
    ):
        """
        Updates position information from IBKR.
        Note: We ignore marketPrice from IBKR and use yfinance instead.

        CRITICAL: Convert IBKR symbol to yfinance format for consistency.
        IBKR returns "8002" but config uses "8002.T", so we need to convert.
        """
        # Convert IBKR symbol to yfinance format using exchange manager
        ibkr_symbol = contract.symbol
        yfinance_symbol = ibkr_symbol  # Default if no conversion needed

        # AUTHORITATIVE FIRST: if the bare IBKR symbol uniquely maps to a
        # configured ticker, use that config key directly. This bypasses the
        # lossy currency-inference path (EUR -> "SBF"/Paris) that produced
        # wrong ".PA" keys for Milan/Madrid/Vienna/Lisbon tickers. Only
        # configured, non-ambiguous symbols are resolved here; everything
        # else falls through to the existing exchange/currency logic below.
        mapped = self._bare_to_config.get(ibkr_symbol)
        if mapped:
            yfinance_symbol = mapped
        elif self.exchange_manager:
            # First, try to use contract.exchange if available
            exchange = contract.exchange if contract.exchange else None

            # If exchange is empty or None, try to infer from currency
            if not exchange or exchange == "":
                currency = getattr(contract, "currency", "USD")
                # Map currency to exchange
                currency_to_exchange = {
                    "JPY": "TSEJ",
                    "GBP": "LSE",
                    "HKD": "SEHK",
                    "AUD": "ASX",
                    "EUR": "SBF",  # or IBIS for German stocks
                    "INR": "NSE",  # India - National Stock Exchange
                    "USD": "SMART",
                }
                exchange = currency_to_exchange.get(currency, "SMART")
                self.logger.debug(
                    f"Inferred exchange {exchange} from currency {currency} for {ibkr_symbol}"
                )

            # Final fallback: check if ibkr_symbol with suffix exists in SYMBOLS
            if not exchange or exchange == "SMART":
                # Try to match against SYMBOLS list
                for symbol in SYMBOLS:
                    if symbol.startswith(ibkr_symbol) and "." in symbol:
                        # Found a match like "8002.T" for "8002"
                        yfinance_symbol = symbol
                        self.logger.info(
                            f"Matched {ibkr_symbol} to {yfinance_symbol} from SYMBOLS list"
                        )
                        break
                else:
                    # No match found, use exchange conversion
                    yfinance_symbol = self.exchange_manager.ibkr_to_yfinance_symbol(
                        ibkr_symbol, exchange
                    )
            else:
                # Convert from IBKR format to yfinance format
                # Uses ibkr_to_yfinance_symbol which handles special cases like:
                # - RR. on LSE -> RR.L (Rolls-Royce)
                # - BA. on LSE -> BA.L (BAE Systems)
                # - 8002 on TSEJ -> 8002.T (standard suffix)
                yfinance_symbol = self.exchange_manager.ibkr_to_yfinance_symbol(
                    ibkr_symbol, exchange
                )

        if position == 0:
            # CRITICAL: During initial position load, ignore position=0 callbacks
            # IBKR sends position=0 for closed positions - these would clear the dict!
            # Both updatePortfolio() and position() can send position=0 during initial load
            if self.initial_position_load_mode:
                self.logger.debug(
                    f"Ignoring position=0 for {yfinance_symbol} during initial load mode"
                )
                return  # DON'T remove during initial load!

            # Normal operation - position was actually closed, remove it
            self.current_positions.pop(yfinance_symbol, None)
            self.current_positions.pop(ibkr_symbol, None)  # Cleanup old format
            self.logger.info(
                f"Position closed: {yfinance_symbol} (removed from tracking)"
            )
        else:
            self.current_positions[yfinance_symbol] = {
                "contract": contract,
                "position": position,
                "averageCost": averageCost,
                "unrealizedPNL": unrealizedPNL,
                "realizedPNL": realizedPNL,
                "accountName": accountName,
            }
            # CRITICAL FIX: Only remove old IBKR format if keys are DIFFERENT
            # For US stocks: yfinance_symbol == ibkr_symbol (both "AVGO")
            # Popping would delete what we just stored!
            if yfinance_symbol != ibkr_symbol:
                self.current_positions.pop(ibkr_symbol, None)

        self.logger.info(
            f"Updated position for {ibkr_symbol} (stored as {yfinance_symbol}): {position} shares"
        )

        # Save state for oversight system
        self._save_state_for_oversight()

    def reconcile_position_keys(self, *, dry_run: bool = False) -> list[dict]:
        """Merge stray exchange-key duplicates into their canonical config key.

        The historical EUR->"SBF" inference bug stored some European
        instruments under a wrong ".PA" key (e.g. ``XYZ.PA``) alongside or
        instead of the correct config key (``XYZ.MI``). This reconciler
        detects any stored key whose bare symbol maps to a configured ticker
        under a DIFFERENT suffix and collapses it onto the canonical key.

        Resolution rule (same instrument, so do NOT sum):
          * If both the canonical and the stray key exist, keep the canonical
            entry (it reflects the latest authoritative resolution) and drop
            the stray. If only the stray exists, RENAME it to canonical.

        Idempotent: a second pass finds nothing to do.

        Args:
            dry_run: if True, report what WOULD change without mutating state.

        Returns:
            A list of change records: {bare, stray_key, canonical_key, action}.
        """
        changes: list[dict] = []
        # Snapshot keys to mutate dict safely during iteration.
        for stored_key in list(self.current_positions.keys()):
            bare = stored_key.split(".")[0]
            canonical = self._bare_to_config.get(bare)
            # Only act when (a) the bare symbol uniquely maps to a config key,
            # and (b) the stored key differs from that canonical key.
            if not canonical or stored_key == canonical:
                continue
            if canonical in self.current_positions:
                action = "drop_stray_duplicate"
                if not dry_run:
                    self.current_positions.pop(stored_key, None)
            else:
                action = "rename_to_canonical"
                if not dry_run:
                    self.current_positions[canonical] = self.current_positions.pop(stored_key)
            changes.append({
                "bare": bare,
                "stray_key": stored_key,
                "canonical_key": canonical,
                "action": action,
            })
            self.logger.warning(
                "reconcile_position_keys: %s stray=%s -> canonical=%s",
                action, stored_key, canonical,
            )
        if changes and not dry_run:
            self._save_state_for_oversight()
        if not changes:
            self.logger.info("reconcile_position_keys: no stray keys found (clean).")
        return changes

    def get_account_value(self, key):
        """Retrieves a specific account value."""
        return self.account_values.get(key, {}).get("value")

    def get_current_net_liquidation(self):
        """Returns the current Net Liquidation value as a float."""
        value = self.get_account_value("NetLiquidation")
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            self.logger.error(f"Invalid NetLiquidation value: {value}")
            return 0.0

    def _calculate_target_value(self, symbol: str, net_liq: float) -> float:
        """
        Calculate target position value based on leverage mode.

        Args:
            symbol: Ticker symbol
            net_liq: Net liquidation value

        Returns:
            Target position value in dollars
        """
        from config import LEVERAGE_MODE, GENERAL_LEVERAGE, ASSET_SPECIFIC_CONFIGS

        portfolio_weight = self.target_allocation.get(symbol, 0.0)

        if LEVERAGE_MODE == "portfolio_mode":
            # NEW LOGIC: Scale entire account then allocate
            leveraged_capital = net_liq * GENERAL_LEVERAGE
            target_value = leveraged_capital * portfolio_weight

            self.logger.info(f"[PORTFOLIO MODE] {symbol}:")
            self.logger.info(f"  Net Liq: ${net_liq:,.2f}")
            self.logger.info(f"  General Leverage: {GENERAL_LEVERAGE}x")
            self.logger.info(f"  Leveraged Capital: ${leveraged_capital:,.2f}")
            self.logger.info(f"  Portfolio Weight: {portfolio_weight:.1%}")
            self.logger.info(f"  Target Value: ${target_value:,.2f}")

        elif LEVERAGE_MODE == "isolated_mode":
            # OLD LOGIC: Per-ticker Kelly fraction
            kelly_fraction = ASSET_SPECIFIC_CONFIGS[symbol].get("kelly_fraction", 1.0)
            target_value = net_liq * portfolio_weight * kelly_fraction

            self.logger.info(f"[ISOLATED MODE] {symbol}:")
            self.logger.info(f"  Net Liq: ${net_liq:,.2f}")
            self.logger.info(f"  Portfolio Weight: {portfolio_weight:.1%}")
            self.logger.info(f"  Kelly Fraction: {kelly_fraction}x")
            self.logger.info(f"  Target Value: ${target_value:,.2f}")

        else:
            raise ValueError(
                f"Invalid LEVERAGE_MODE: {LEVERAGE_MODE}. Must be 'portfolio_mode' or 'isolated_mode'"
            )

        return target_value

    def _fetch_yfinance_price(self, symbol, max_retries=3):
        """
        Fetch latest close price from yfinance with retry logic.

        Args:
            symbol: Ticker symbol.
            max_retries: Number of retry attempts on failure.

        Returns:
            Tuple of (hist_dataframe, repair_flag) or (None, None) if all retries fail.
        """
        import yfinance as yf

        repair = symbol.endswith(".L")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        for attempt in range(1, max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, repair=repair)
                if not hist.empty:
                    # Drop rows where Close is NaN (common for current trading day
                    # before market close, or partial holiday rows from yfinance)
                    hist = hist.dropna(subset=["Close"])
                    if not hist.empty:
                        return hist, repair
                    else:
                        self.logger.warning(
                            f"{symbol}: yfinance returned data but all Close values are NaN "
                            f"(attempt {attempt}/{max_retries})"
                        )
                else:
                    self.logger.warning(
                        f"{symbol}: yfinance returned empty data (attempt {attempt}/{max_retries})"
                    )
            except Exception as e:
                self.logger.warning(
                    f"{symbol}: yfinance error (attempt {attempt}/{max_retries}): {e}"
                )

            # Exponential backoff: 3s, 6s, 10s
            if attempt < max_retries:
                backoff = 3 * attempt + random.uniform(0, 1)
                self.logger.debug(f"{symbol}: Retrying in {backoff:.1f}s...")
                time.sleep(backoff)

        return None, repair

    def _fetch_parquet_fallback_price(self, symbol):
        """
        Fallback: Get latest close price from the parquet store / data_manager cache.

        The parquet data is already loaded by data_manager during signal generation.
        This avoids skipping a symbol just because yfinance had a transient outage.

        Returns:
            Tuple of (close_price, price_date) or (None, None) if unavailable.
        """
        # Try data_manager's cached historical data first (already in memory)
        if self.data_manager is not None:
            hist_df = self.data_manager.historical_data.get(symbol)
            if hist_df is not None and not hist_df.empty and "Close" in hist_df.columns:
                latest_close = hist_df["Close"].iloc[-1]
                latest_date = hist_df.index[-1]
                if latest_close > 0:
                    return float(latest_close), latest_date

        # Try parquet store directly
        try:
            from algos.common.market_data_store import MarketDataStore

            store = MarketDataStore()
            if store.has_ticker(symbol):
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
                df = store.get_ohlcv(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    close_col = "Close" if "Close" in df.columns else "close"
                    if close_col in df.columns:
                        latest_close = df[close_col].iloc[-1]
                        latest_date = df.index[-1]
                        if latest_close > 0:
                            return float(latest_close), latest_date
        except Exception as e:
            self.logger.debug(f"{symbol}: Parquet store fallback error: {e}")

        return None, None

    def _fetch_ibkr_snapshot_fallback_price(self, symbol):
        """
        Last resort fallback: Get price from IBKR market data snapshot.

        Requires an active IBKR connection and the market to be open (or have
        a valid close price). Uses market_data_manager.request_snapshot().

        Returns:
            float price or None if unavailable.
        """
        if self.market_data_manager is None or self.contract_details_mgr is None:
            return None

        try:
            # Build a basic contract for the symbol
            contract = Contract()
            details = self.contract_details_mgr.get_details(symbol)
            if not details:
                return None

            contract.symbol = details.get("local_symbol", symbol.split(".")[0])
            contract.secType = "STK"
            contract.exchange = details.get("exchange", "SMART")
            contract.currency = details.get("currency", "USD")

            snapshot = self.market_data_manager.request_snapshot(
                symbol, contract, timeout=8.0
            )
            if snapshot is None:
                return None

            # Prefer last trade, then close, then midpoint
            price = snapshot.get("last") or snapshot.get("close")
            if price is None:
                bid = snapshot.get("bid")
                ask = snapshot.get("ask")
                if bid and ask and bid > 0 and ask > 0:
                    price = (bid + ask) / 2.0

            if price and price > 0:
                return float(price)
        except Exception as e:
            self.logger.debug(f"{symbol}: IBKR snapshot fallback error: {e}")

        return None

    def _store_price(
        self, symbol, latest_close, repair, price_date=None, source="yfinance"
    ):
        """
        Apply magnifier correction, convert to USD, and store price in caches.

        Args:
            symbol: Ticker symbol.
            latest_close: Native currency close price.
            repair: Whether yfinance repair mode was used (skip magnifier if True).
            price_date: Date of the price (for logging). Optional.
            source: Price source label for logging.

        Returns:
            True if price was stored successfully, False otherwise.
        """
        # Reject NaN prices at the gate — they must never enter the price cache
        import math

        if latest_close is None or math.isnan(latest_close):
            self.logger.error(
                f"{symbol}: Rejecting NaN/None price from {source}. "
                f"Price will NOT be stored."
            )
            return False

        # Apply priceMagnifier correction (skip if repair mode already handled it)
        if self.contract_details_mgr and not repair:
            details = self.contract_details_mgr.get_details(symbol)
            magnifier = details.get("priceMagnifier", 1)

            if magnifier > 1:
                corrected_price = latest_close / magnifier
                self.logger.info(
                    f"{symbol}: Applied magnifier correction /{magnifier}: {latest_close:.4f} -> {corrected_price:.4f}"
                )
                latest_close = corrected_price
        elif repair:
            self.logger.debug(
                f"{symbol}: Using repair=True, price already in base currency (no magnifier needed)"
            )

        # Get currency from exchange manager
        currency = "USD"
        if self.exchange_manager:
            currency = self.exchange_manager.get_currency(symbol)

        # Store native currency price
        self.latest_prices_native[symbol] = float(latest_close)
        self.price_currencies[symbol] = currency

        # Convert to USD if needed
        if currency != "USD" and self.currency_converter:
            usd_price = self.currency_converter.convert_to_usd(latest_close, currency)
            if usd_price and usd_price > 0:
                self.logger.info(
                    f"{symbol}: Converted {latest_close:.4f} {currency} -> ${usd_price:.4f} USD"
                )
                self.latest_prices_usd[symbol] = float(usd_price)
            else:
                self.logger.error(
                    f"{symbol}: Failed to convert {latest_close:.4f} {currency} to USD (rate invalid or unavailable). Skipping."
                )
                return False
        else:
            self.latest_prices_usd[symbol] = float(latest_close)

        # Log price info
        date_str = ""
        if price_date is not None:
            day_name = (
                price_date.strftime("%A")
                if hasattr(price_date, "strftime")
                else "Unknown"
            )
            date_label = (
                price_date.date() if hasattr(price_date, "date") else price_date
            )
            date_str = f"Date={date_label} ({day_name}), "

        self.logger.info(
            f"{symbol}: {date_str}Native={latest_close:.4f} {currency}, "
            f"USD=${self.latest_prices_usd[symbol]:.4f} [source: {source}]"
        )
        return True

    def fetch_latest_prices(self, force_update=False):
        """
        Fetches latest EOD prices for all symbols with a multi-source fallback chain:
          1. IBKR historical bar (primary, authoritative) — via ibkr_data_manager
          2. Parquet store / data_manager cache (stale fallback)
          3. IBKR live market data snapshot (requires market open)
          4. yfinance (last resort for uncovered symbols only)

        Multi-currency support:
        - Applies repair=True for LSE stocks to fix GBX/GBP issues
        - Applies priceMagnifier correction from IBKR contract details
        - Converts foreign currency prices to USD for position sizing
        """
        now = datetime.now()

        # Use cache if recent (within 1 minute) unless forced
        if not force_update and self.last_price_update:
            if (now - self.last_price_update).total_seconds() < 60:
                self.logger.debug("Using cached prices (less than 60 seconds old)")
                return self.latest_prices

        self.logger.info(
            "Fetching latest prices (IBKR -> parquet -> snapshot -> yfinance fallback)..."
        )

        for symbol in SYMBOLS:
            # Skip blacklisted symbols
            if symbol in BLACKLISTED_SYMBOLS:
                self.logger.debug(
                    f"Skipping price fetch for blacklisted symbol: {symbol}"
                )
                continue

            # === Source 1: IBKR historical bar (primary, authoritative) ===
            if self.ibkr_data_manager is not None:
                result = self.ibkr_data_manager.fetch_latest_close(symbol)
                if result is not None:
                    ibkr_close, ibkr_date = result
                    repair_flag = symbol.endswith(".L")
                    stored = self._store_price(
                        symbol, ibkr_close, repair_flag, ibkr_date, source="ibkr"
                    )
                    if stored:
                        time.sleep(random.uniform(0.3, 0.6))
                        continue

            # === Source 2: Parquet store / data_manager cache ===
            self.logger.warning(
                f"{symbol}: IBKR price unavailable. Trying parquet fallback..."
            )
            parquet_price, parquet_date = self._fetch_parquet_fallback_price(symbol)
            if parquet_price is not None:
                # Check staleness (warn if price is more than 3 trading days old)
                if parquet_date is not None:
                    try:
                        date_obj = (
                            parquet_date.date()
                            if hasattr(parquet_date, "date")
                            else parquet_date
                        )
                        days_old = (datetime.now().date() - date_obj).days
                        if days_old > 5:
                            self.logger.warning(
                                f"{symbol}: Parquet fallback price is {days_old} days old. Use with caution."
                            )
                    except Exception:
                        pass

                repair_flag = symbol.endswith(".L")
                stored = self._store_price(
                    symbol,
                    parquet_price,
                    repair_flag,
                    parquet_date,
                    source="parquet-fallback",
                )
                time.sleep(random.uniform(0.5, 1.0))
                if stored:
                    continue
                # Fall through to IBKR snapshot

            # === Source 3: IBKR live market data snapshot (requires market open) ===
            self.logger.warning(
                f"{symbol}: Parquet unavailable. Trying IBKR live snapshot..."
            )
            ibkr_price = self._fetch_ibkr_snapshot_fallback_price(symbol)
            if ibkr_price is not None:
                repair_flag = symbol.endswith(".L")
                stored = self._store_price(
                    symbol, ibkr_price, repair_flag, source="ibkr-snapshot"
                )
                if stored:
                    time.sleep(random.uniform(0.5, 1.0))
                    continue

            # === Source 4: yfinance (last resort for uncovered symbols) ===
            self.logger.warning(
                f"{symbol}: IBKR/parquet/snapshot unavailable. Trying yfinance (last resort)..."
            )
            hist, repair = self._fetch_yfinance_price(symbol, max_retries=3)
            if hist is not None:
                latest_close = hist["Close"].iloc[-1]
                price_date = hist.index[-1]
                stored = self._store_price(
                    symbol, latest_close, repair, price_date, source="yfinance"
                )
                time.sleep(random.uniform(1.5, 2.5))
                if stored:
                    continue

            # === All sources exhausted ===
            self.logger.error(
                f"{symbol}: ALL price sources failed "
                f"(IBKR, parquet, IBKR-snapshot, yfinance). "
                f"Symbol will be SKIPPED for this trading cycle."
            )

        self.last_price_update = now
        return self.latest_prices

    def get_price_for_symbol(self, symbol):
        """
        Gets the latest USD price for a symbol (for position sizing).
        Fetches from yfinance if not cached.
        """
        if symbol not in self.latest_prices_usd:
            self.fetch_latest_prices()

        return self.latest_prices_usd.get(symbol, 0.0)

    def get_native_price(self, symbol):
        """
        Gets the latest price in native currency (for order execution).
        Fetches from yfinance if not cached.
        """
        if symbol not in self.latest_prices_native:
            self.fetch_latest_prices()

        return self.latest_prices_native.get(symbol, 0.0)

    def get_price_currency(self, symbol):
        """Get the currency for a symbol's price."""
        return self.price_currencies.get(symbol, "USD")

    def calculate_current_portfolio_metrics(self):
        """
        Calculates current weights and leverage using yfinance prices.
        Properly excludes blacklisted symbols from all calculations.
        """
        net_liq = self.get_current_net_liquidation()
        if not net_liq or net_liq <= 0:
            self.logger.warning("Net Liquidation is zero or negative.")
            return {}, 0.0, 0.0

        # Ensure we have latest prices
        self.fetch_latest_prices()

        current_weights = {}
        gross_exposure = 0.0

        # Process only non-blacklisted symbols
        for symbol in SYMBOLS:
            # Skip blacklisted symbols completely
            if symbol in BLACKLISTED_SYMBOLS:
                self.logger.debug(
                    f"Skipping blacklisted symbol {symbol} in portfolio metrics"
                )
                continue
            # Get position data
            pos_data = self.current_positions.get(symbol, {"position": 0})
            current_shares = float(
                pos_data.get("position", 0)
            )  # Convert to float to handle Decimal

            # Get price from yfinance
            price = float(self.latest_prices.get(symbol, 0.0))  # Ensure float

            if price > 0:
                market_value = current_shares * price
                current_weights[symbol] = market_value / net_liq
                gross_exposure += abs(market_value)

                if current_shares != 0:
                    self.logger.debug(
                        f"{symbol}: {current_shares} shares @ ${price:.2f} = "
                        f"${market_value:.2f} ({current_weights[symbol] * 100:.2f}%)"
                    )
            else:
                self.logger.warning(f"No valid price for {symbol}")

        current_leverage = gross_exposure / net_liq if net_liq > 0 else 0.0

        self.logger.info(
            f"Portfolio Metrics - Net Liq: ${net_liq:,.2f}, "
            f"Gross Exposure: ${gross_exposure:,.2f}, "
            f"Leverage: {current_leverage:.2f}x"
        )

        return current_weights, gross_exposure, current_leverage

    def _get_unwanted_positions(self):
        """
        Get positions that should be closed:
        1. Symbols not in SYMBOLS list (not configured)
        2. Symbols in BLACKLISTED_SYMBOLS

        Deduplicates positions: IBKR reports the same security on multiple exchanges
        (e.g., CLS.TO on TSE and CLS on SMART). Without dedup, both get sold —
        the real position closes correctly but the duplicate creates an unintended short.

        Returns list of symbols to close (deduplicated).
        """
        unwanted = []

        # Known exchange suffixes for dedup grouping
        EXCHANGE_SUFFIXES = (
            ".TO",
            ".V",
            ".MI",
            ".PA",
            ".MC",
            ".BD",
            ".TA",
            ".ST",
            ".L",
            ".LS",
            ".PR",
            ".F",
            ".DE",
            ".AS",
            ".BR",
            ".HE",
            ".CO",
            ".OL",
            ".HK",
            ".SI",
            ".AX",
            ".NZ",
            ".TL",
        )

        def _base_symbol(sym: str) -> str:
            """Strip exchange suffix to get base symbol for dedup."""
            for sfx in EXCHANGE_SUFFIXES:
                if sym.endswith(sfx):
                    return sym[: -len(sfx)]
            # Also handle IBKR's dash-suffix format (e.g., CSH-UN.TO -> CSH-UN -> CSH)
            return sym.split(".")[0]

        # Pass 1: Collect all unconfigured/blacklisted positions
        raw_unwanted = {}  # symbol -> position
        for symbol, pos_data in self.current_positions.items():
            position = float(pos_data.get("position", 0))
            if position == 0:
                continue

            if symbol in BLACKLISTED_SYMBOLS:
                self.logger.info(
                    f"Found blacklisted position: {symbol} ({position} shares)"
                )
                raw_unwanted[symbol] = position
            elif symbol not in SYMBOLS:
                self.logger.info(
                    f"Found unconfigured position: {symbol} ({position} shares)"
                )
                raw_unwanted[symbol] = position

        # Pass 2: Deduplicate — group by base symbol
        # For each group, only keep the PRIMARY entry (the one with an exchange
        # suffix, or if none has a suffix, the one with the larger position).
        # SMART-exchange duplicates (no suffix) are skipped when a suffixed
        # entry exists for the same base symbol.
        from collections import defaultdict

        base_groups = defaultdict(list)
        for sym, pos in raw_unwanted.items():
            base = _base_symbol(sym)
            has_suffix = any(sym.endswith(sfx) for sfx in EXCHANGE_SUFFIXES)
            base_groups[base].append((sym, pos, has_suffix))

        for base, entries in base_groups.items():
            if len(entries) == 1:
                # No duplicate — keep as-is
                unwanted.append(entries[0][0])
            else:
                # Duplicate detected — pick the primary (suffixed) entry
                suffixed = [e for e in entries if e[2]]
                unsuffixed = [e for e in entries if not e[2]]

                if suffixed:
                    # Use the suffixed entry (real exchange position)
                    primary = suffixed[0]
                    unwanted.append(primary[0])
                    for dup in suffixed[1:] + unsuffixed:
                        self.logger.warning(
                            f"Skipping duplicate position {dup[0]} ({dup[1]} shares) "
                            f"— already closing via {primary[0]} ({primary[1]} shares)"
                        )
                else:
                    # No suffixed entry — use the one with largest abs position
                    entries_sorted = sorted(
                        entries, key=lambda e: abs(e[1]), reverse=True
                    )
                    unwanted.append(entries_sorted[0][0])
                    for dup in entries_sorted[1:]:
                        self.logger.warning(
                            f"Skipping duplicate position {dup[0]} ({dup[1]} shares) "
                            f"— already closing via {entries_sorted[0][0]} ({entries_sorted[0][1]} shares)"
                        )

        return unwanted

    def _calculate_available_capital(self, reserved_for_closes=0):
        """
        Calculate available capital for new positions.
        Considers margin requirements and reserved capital for closing positions.
        """
        net_liq = self.get_current_net_liquidation()
        if not net_liq:
            return 0

        # Get available funds (considers margin)
        available_funds_str = self.account_values.get("AvailableFunds", {}).get(
            "value", str(net_liq)
        )
        try:
            available_funds = float(available_funds_str)
        except (ValueError, TypeError):
            available_funds = net_liq

        # Reserve capital for closing unwanted positions
        available_capital = available_funds - reserved_for_closes

        # Safety margin - keep 5% buffer
        safety_buffer = net_liq * 0.05
        available_capital = max(0, available_capital - safety_buffer)

        return available_capital

    def get_trades_for_signal_based_execution(
        self, signals: Dict, symbols_to_trade: list = None
    ):
        """
        Determines trades based on strategy type:
        - ml_signal: Use ML predictions (signal +1/-1/0)
        - buy_and_hold: Always maintain target position (treat as signal +1)

        Rebalancing and position sizing apply to both strategies.
        Returns {symbol: shares_to_trade} where positive = BUY, negative = SELL.

        Args:
            signals: Dict of {symbol: signal} from ML models
            symbols_to_trade: List of symbols to trade (default: all SYMBOLS from config)
                             Use this to filter by region when running regional sessions.
        """
        # Log strategy mode once
        if not getattr(self, "_strategy_mode_logged", False):
            try:
                from config import STRATEGY_MODE

                self.logger.info(f"[PortfolioManager] Strategy mode: {STRATEGY_MODE}")
            except ImportError:
                pass
            self._strategy_mode_logged = True

        net_liq = self.get_current_net_liquidation()
        if not net_liq or net_liq <= 0:
            self.logger.error("Net Liquidation not available. Cannot trade.")
            return {}

        # Get latest prices from yfinance
        self.fetch_latest_prices(force_update=True)

        from config import LEVERAGE_MODE, DEFAULT_STRATEGY_TYPE

        self.logger.info("=" * 50)
        self.logger.info(f"SIGNAL-BASED TRADING - {LEVERAGE_MODE.upper()}")
        self.logger.info(f"Net Liquidation: ${net_liq:,.2f}")

        # CRITICAL SAFETY CHECK: Log current positions before trading
        # This helps detect position loading failures
        current_position_count = sum(
            1
            for pos in self.current_positions.values()
            if float(pos.get("position", 0)) != 0
        )

        self.logger.info(f"Current positions before trading: {current_position_count}")
        if current_position_count > 0:
            for symbol, pos_data in self.current_positions.items():
                position = float(pos_data.get("position", 0))
                if position != 0:
                    self.logger.info(f"  {symbol}: {position} shares")
        else:
            self.logger.warning(
                "No current positions detected - all positions will show as 0.0 shares"
            )
            self.logger.warning(
                "If you expect positions, CHECK POSITION LOADING ABOVE!"
            )

        trades_needed = {}

        # Step 1: Close unwanted positions first (blacklisted or unconfigured)
        # IMPORTANT: Respect minimum positions even for unwanted symbols
        unwanted_positions = self._get_unwanted_positions()
        for symbol in unwanted_positions:
            pos_data = self.current_positions.get(symbol, {"position": 0})
            current_shares = float(pos_data.get("position", 0))
            if current_shares != 0:
                # Check if minimum position is configured (even for blacklisted symbols)
                asset_config = self.asset_configs.get(symbol, {})
                min_shares = asset_config.get("min_position_shares", None)

                if min_shares is not None and min_shares > 0:
                    # Keep minimum position, sell the rest
                    if current_shares > min_shares:
                        shares_to_sell = current_shares - min_shares
                        trades_needed[symbol] = -shares_to_sell
                        self.logger.info(
                            f"Closing unwanted position (keeping minimum): {symbol} - SELL {abs(shares_to_sell)} shares (keep {min_shares})"
                        )
                    elif current_shares == min_shares:
                        # Already at minimum, keep it
                        self.logger.info(
                            f"Unwanted position at minimum: {symbol} - keeping {current_shares} shares (min={min_shares})"
                        )
                    else:
                        # Below minimum, buy up to minimum even for unwanted positions
                        shares_to_buy = min_shares - current_shares
                        trades_needed[symbol] = shares_to_buy
                        self.logger.info(
                            f"Unwanted position below minimum: {symbol} - BUY {shares_to_buy} shares to reach minimum {min_shares}"
                        )
                else:
                    # No minimum, close completely
                    trades_needed[symbol] = -current_shares  # Sell all shares
                    self.logger.info(
                        f"Closing unwanted position: {symbol} - SELL {abs(current_shares)} shares"
                    )

        # Step 2: Process signals for configured symbols
        # Use provided list or default to all SYMBOLS
        trading_symbols = symbols_to_trade if symbols_to_trade is not None else SYMBOLS
        for symbol in trading_symbols:
            # Skip blacklisted symbols entirely
            if symbol in BLACKLISTED_SYMBOLS:
                self.logger.info(f"Skipping blacklisted symbol: {symbol}")
                continue

            # Get strategy type for this ticker
            asset_config = self.asset_configs.get(symbol, {})
            strategy_type = asset_config.get("strategy_type", DEFAULT_STRATEGY_TYPE)

            # Get current position
            pos_data = self.current_positions.get(symbol, {"position": 0})
            current_shares = float(pos_data.get("position", 0))

            # Get USD-converted price (self.latest_prices is an alias for self.latest_prices_usd)
            current_price = float(self.latest_prices.get(symbol, 0.0))

            # NaN comparisons always return False, so 'NaN <= 0' is False.
            # Must check explicitly with math.isnan or not-greater-than-zero.
            import math

            if current_price <= 0 or math.isnan(current_price):
                # Price may be missing due to yfinance failure OR currency conversion failure
                currency = self.price_currencies.get(symbol, "USD")
                if currency == "USD":
                    self.logger.error(
                        f"Cannot trade {symbol}: yfinance returned no data or invalid price"
                    )
                else:
                    self.logger.error(
                        f"Cannot trade {symbol}: Currency conversion failed ({currency} -> USD)"
                    )
                continue

            self.logger.info(f"\n{symbol} Trading ({strategy_type}):")

            # Determine effective signal based on strategy
            if strategy_type == "buy_and_hold":
                # Buy-and-hold: Always treat as BUY signal (signal = 1)
                effective_signal = 1
                self.logger.info(f"  Strategy: BUY-AND-HOLD (always hold position)")
                self.logger.info(
                    f"  Current position: {current_shares} shares @ ${current_price:.2f}"
                )

            elif strategy_type == "ml_signal":
                # ML-based: Use signal from ML model
                if symbol not in signals:
                    self.logger.warning(f"  No ML signal for {symbol}. Skipping.")
                    continue

                effective_signal = signals[symbol]
                self.logger.info(f"  Strategy: ML-SIGNAL")
                self.logger.info(f"  Signal: {effective_signal}")
                self.logger.info(
                    f"  Current position: {current_shares} shares @ ${current_price:.2f}"
                )

            else:
                self.logger.error(
                    f"  Unknown strategy_type for {symbol}: {strategy_type}"
                )
                continue

            # Execute trades based on effective signal
            if effective_signal == 1:
                # BUY SIGNAL (or buy-and-hold maintenance)
                # Skip blacklisted symbols
                if symbol in BLACKLISTED_SYMBOLS:
                    self.logger.warning(
                        f"  {symbol} is blacklisted. Skipping all trades."
                    )
                    continue

                if symbol not in self.target_allocation:
                    self.logger.warning(
                        f"  {symbol} not in TARGET_ALLOCATION. Skipping."
                    )
                    continue

                if symbol not in self.asset_configs:
                    self.logger.error(
                        f"  {symbol} not in ASSET_SPECIFIC_CONFIGS. Skipping."
                    )
                    continue

                # Use unified calculation method (supports both leverage modes)
                target_value = self._calculate_target_value(symbol, net_liq)

                # Calculate current position value and shares to trade
                current_position_value = current_shares * current_price

                # Calculate target shares from target value
                target_shares = round(target_value / current_price)

                # Apply minimum position floor if configured
                min_shares = asset_config.get("min_position_shares", None)
                if min_shares is not None and min_shares > 0:
                    if target_shares < min_shares:
                        self.logger.info(
                            f"  Applying minimum position floor: {min_shares} shares (calculated: {target_shares})"
                        )
                        target_shares = min_shares

                shares_to_trade = target_shares - current_shares

                self.logger.info(f"  Current Position:")
                self.logger.info(f"    Current Shares: {current_shares}")
                self.logger.info(f"    Current Value: ${current_position_value:,.2f}")
                self.logger.info(f"  Target Position:")
                self.logger.info(f"    Target Shares: {target_shares}")
                self.logger.info(f"    Target Value: ${target_value:,.2f}")
                self.logger.info(f"  Trade Required:")
                self.logger.info(f"    Shares to Trade: {shares_to_trade}")

                if abs(shares_to_trade) >= MIN_TRADE_SHARES:
                    trades_needed[symbol] = shares_to_trade
                    if strategy_type == "buy_and_hold":
                        action = "REBALANCE (buy-and-hold)"
                    elif shares_to_trade > 0:
                        action = "BUY" if current_shares == 0 else "ADD"
                    else:
                        action = "REDUCE"
                    self.logger.info(
                        f"  -> ACTION: {action} {abs(shares_to_trade)} shares"
                    )
                else:
                    self.logger.info(
                        f"  -> No trade: Already at target or change too small"
                    )

            elif effective_signal == -1:
                # SELL SIGNAL - respect minimum position if configured
                min_shares = asset_config.get("min_position_shares", None)

                if min_shares is not None and min_shares > 0:
                    # Minimum position configured - keep minimum, sell the rest
                    if current_shares > min_shares:
                        # Sell down to minimum
                        shares_to_sell = current_shares - min_shares
                        trades_needed[symbol] = -shares_to_sell
                        self.logger.info(
                            f"  Minimum Position: {min_shares} shares (configured)"
                        )
                        self.logger.info(
                            f"  -> ACTION: SELL {shares_to_sell} shares (keep minimum {min_shares})"
                        )
                    elif current_shares == min_shares:
                        # Already at minimum, no trade
                        self.logger.info(
                            f"  Minimum Position: {min_shares} shares (configured)"
                        )
                        self.logger.info(
                            f"  -> No trade: Already at minimum position ({min_shares} shares)"
                        )
                    else:
                        # Below minimum on sell signal - BUY up to minimum (overrides sell signal)
                        shares_to_buy = min_shares - current_shares
                        trades_needed[symbol] = shares_to_buy
                        self.logger.info(
                            f"  Minimum Position: {min_shares} shares (configured)"
                        )
                        self.logger.info(
                            f"  -> ACTION: BUY {shares_to_buy} shares (enforce minimum {min_shares}, overriding SELL signal)"
                        )
                else:
                    # No minimum configured - exit completely (original behavior)
                    if current_shares > 0:
                        trades_needed[symbol] = -current_shares
                        self.logger.info(
                            f"  -> ACTION: SELL ALL {current_shares} shares (EXIT POSITION)"
                        )
                    else:
                        self.logger.info(f"  -> No trade: No position to sell")

            else:
                # NEUTRAL SIGNAL (0)
                self.logger.info(f"  -> No trade: Neutral signal")

        self.logger.info("=" * 50)

        if trades_needed:
            self.logger.info(f"TRADES TO EXECUTE: {trades_needed}")

            # Show expected portfolio after trades
            self.logger.info("\nEXPECTED PORTFOLIO AFTER TRADES:")
            total_expected_value = 0
            for symbol, shares_to_trade in trades_needed.items():
                current_pos = float(
                    self.current_positions.get(symbol, {}).get("position", 0)
                )
                expected_pos = current_pos + shares_to_trade
                if expected_pos != 0:
                    price = float(self.latest_prices.get(symbol, 0))
                    value = expected_pos * price
                    total_expected_value += value
                    self.logger.info(
                        f"  {symbol}: {expected_pos} shares (${value:,.2f})"
                    )

            # Show remaining configured symbols not being traded
            for symbol in SYMBOLS:
                if symbol not in trades_needed:
                    current_pos = float(
                        self.current_positions.get(symbol, {}).get("position", 0)
                    )
                    if current_pos != 0:
                        price = float(self.latest_prices.get(symbol, 0))
                        value = current_pos * price
                        total_expected_value += value
                        self.logger.info(
                            f"  {symbol}: {current_pos} shares (${value:,.2f}) - unchanged"
                        )

            expected_leverage = total_expected_value / net_liq if net_liq > 0 else 0
            self.logger.info(
                f"Total expected portfolio value: ${total_expected_value:,.2f}"
            )
            self.logger.info(f"Expected leverage: {expected_leverage:.2f}x")
        else:
            self.logger.info("NO TRADES NEEDED")

        return trades_needed

    def get_trades_for_rebalance(self):
        """
        Determines trades needed for rebalancing using yfinance prices.
        Returns {symbol: shares_to_trade} where positive = BUY, negative = SELL.
        """
        net_liq = self.get_current_net_liquidation()
        if not net_liq or net_liq <= 0:
            self.logger.error("Net Liquidation not available. Cannot rebalance.")
            return {}

        net_liq = float(net_liq)  # Ensure net_liq is a float to handle Decimal

        # Get latest prices from yfinance
        self.fetch_latest_prices(force_update=True)

        # Calculate current portfolio metrics
        current_weights, _, current_leverage = (
            self.calculate_current_portfolio_metrics()
        )

        self.logger.info("=" * 50)
        self.logger.info("REBALANCING CALCULATION")
        self.logger.info(f"Net Liquidation: ${net_liq:,.2f}")
        self.logger.info(f"Current Leverage: {current_leverage:.2f}x")
        self.logger.info(f"Current Weights: {current_weights}")

        trades_needed = {}

        # Check for orphaned positions (positions not in current SYMBOLS)
        for symbol, pos_data in self.current_positions.items():
            if symbol not in SYMBOLS:
                # Skip blacklisted symbols
                if symbol in BLACKLISTED_SYMBOLS:
                    current_shares = float(pos_data.get("position", 0))
                    if current_shares != 0:
                        self.logger.info(
                            f"{symbol} is blacklisted (e.g., promotional shares) - {current_shares} shares will be ignored"
                        )
                    continue

                current_shares = float(pos_data.get("position", 0))
                if current_shares != 0:
                    # Get price for orphaned symbol
                    try:
                        import yfinance as yf

                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="2d")
                        if not hist.empty:
                            price = float(hist["Close"].iloc[-1])
                            self.logger.info(
                                f"{symbol} no longer in config but has {current_shares} shares @ ${price:.2f} - marking for liquidation"
                            )
                        else:
                            self.logger.info(
                                f"{symbol} no longer in config but has {current_shares} shares - marking for liquidation"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not get price for orphaned {symbol}: {e}"
                        )
                        self.logger.info(
                            f"{symbol} has {current_shares} shares - marking for liquidation anyway"
                        )

                    trades_needed[symbol] = -current_shares  # Sell all

        for symbol in SYMBOLS:
            # Skip blacklisted symbols entirely
            if symbol in BLACKLISTED_SYMBOLS:
                self.logger.info(
                    f"Skipping blacklisted symbol in rebalancing: {symbol}"
                )
                continue

            # Check configuration
            if symbol not in self.target_allocation:
                self.logger.warning(f"{symbol} not in TARGET_ALLOCATION. Skipping.")
                continue

            if symbol not in self.asset_configs:
                self.logger.error(f"{symbol} not in ASSET_SPECIFIC_CONFIGS. Skipping.")
                continue

            # Get current position
            pos_data = self.current_positions.get(symbol, {"position": 0})
            current_shares = float(
                pos_data.get("position", 0)
            )  # Convert to float to handle Decimal

            # Get price from yfinance
            current_price = float(self.latest_prices.get(symbol, 0.0))  # Ensure float

            if current_price <= 0:
                self.logger.error(
                    f"Cannot rebalance {symbol}: No valid price from yfinance"
                )
                continue

            # Use unified calculation method (supports both leverage modes)
            self.logger.info(f"\n{symbol} Rebalancing:")
            self.logger.info(
                f"  Current: {current_shares} shares @ ${current_price:.2f}"
            )

            target_value = self._calculate_target_value(symbol, net_liq)

            # Convert to shares
            target_shares = round(target_value / current_price)

            # Calculate trade needed
            shares_to_trade = target_shares - current_shares

            self.logger.info(f"  Target shares: {target_shares}")
            self.logger.info(f"  Shares to trade: {shares_to_trade}")

            # Check if trade is needed
            current_value = current_shares * current_price
            current_weight = current_value / net_liq if net_liq > 0 else 0.0

            # Calculate expected weight from target_value (which already accounts for leverage mode)
            target_weight_actual = target_value / net_liq if net_liq > 0 else 0.0
            weight_deviation = abs(current_weight - target_weight_actual)

            # Decision logic
            need_trade = False

            # Trade if:
            # 1. Deviation is significant
            # 2. We need to exit a position (target=0, current!=0)
            # 3. We need to enter a position (target!=0, current=0)
            # 4. Trade size meets minimum threshold

            if abs(shares_to_trade) >= MIN_TRADE_SHARES:
                if weight_deviation >= REBALANCE_THRESHOLD_PERCENT:
                    need_trade = True
                    self.logger.info(
                        f"  -> Trade needed: Weight deviation {weight_deviation * 100:.2f}%"
                    )
                elif target_shares == 0 and current_shares != 0:
                    need_trade = True
                    self.logger.info(f"  -> Trade needed: Exiting position")
                elif target_shares != 0 and current_shares == 0:
                    need_trade = True
                    self.logger.info(f"  -> Trade needed: Entering new position")
            else:
                self.logger.info(
                    f"  -> No trade: Size below minimum ({abs(shares_to_trade)} < {MIN_TRADE_SHARES})"
                )

            if need_trade:
                trades_needed[symbol] = shares_to_trade
                action = "BUY" if shares_to_trade > 0 else "SELL"
                self.logger.info(f"  -> ACTION: {action} {abs(shares_to_trade)} shares")
            else:
                if weight_deviation < REBALANCE_THRESHOLD_PERCENT:
                    self.logger.info(f"  -> No trade: Within threshold")

        self.logger.info("=" * 50)

        if trades_needed:
            self.logger.info(f"TRADES TO EXECUTE: {trades_needed}")
        else:
            self.logger.info("NO TRADES NEEDED")

        return trades_needed

    def _save_state_for_oversight(self):
        """Save current state for portfolio oversight system.

        GUARD: never overwrite a populated live state file with an EMPTY
        in-memory state. A non-live PortfolioManager (e.g. a unit test, a
        reconciler demo, or a partially-initialised instance) has empty
        ``account_values``/``current_positions``; persisting that would clobber
        the live NAV cache and break the next cron's NAV gate (this happened
        2026-06-09, see MEMORY.md "account_values.pkl clobbered to empty").
        An account genuinely going to zero is not represented by an empty dict
        — it is a NetLiquidation value of 0 — so this guard cannot mask a real
        emptying event.
        """
        try:
            # Save account values as pickle — but never write an empty dict over
            # an existing populated file.
            if self.account_values or not self._file_has_data(self.account_state_file):
                self._atomic_pickle(self.account_state_file, self.account_values)
            else:
                self.logger.debug(
                    "Skipping account_values save: in-memory state empty but "
                    "%s has data (refusing to clobber live NAV cache).",
                    self.account_state_file.name,
                )

            # Save positions — same guard.
            if self.current_positions or not self._file_has_data(self.positions_file):
                self._atomic_pickle(self.positions_file, self.current_positions)
            else:
                self.logger.debug(
                    "Skipping positions save: in-memory state empty but %s has data.",
                    self.positions_file.name,
                )

            # Also save as JSON for readability (need to serialize Contract objects)
            positions_serializable = {}
            for symbol, pos_data in self.current_positions.items():
                # Create serializable version without Contract object
                serializable_pos = {
                    "position": float(pos_data.get("position", 0)),
                    "averageCost": float(pos_data.get("averageCost", 0)),
                    "unrealizedPNL": float(pos_data.get("unrealizedPNL", 0)),
                    "realizedPNL": float(pos_data.get("realizedPNL", 0)),
                    "accountName": str(pos_data.get("accountName", "")),
                }
                # Add contract info if available
                if "contract" in pos_data and pos_data["contract"]:
                    contract = pos_data["contract"]
                    serializable_pos["contract_info"] = {
                        "symbol": getattr(contract, "symbol", symbol),
                        "exchange": getattr(contract, "exchange", "SMART"),
                        "currency": getattr(contract, "currency", "USD"),
                    }
                positions_serializable[symbol] = serializable_pos

            # JSON mirror — skip if we have nothing to write but a populated
            # JSON already exists (same anti-clobber rule as the pickles).
            if (self.account_values or self.current_positions
                    or not self.json_state_file.exists()):
                state = {
                    "timestamp": datetime.now().isoformat(),
                    "account_values": {k: v for k, v in self.account_values.items()},
                    "positions": positions_serializable,
                }
                tmp = self.json_state_file.with_suffix(self.json_state_file.suffix + ".tmp")
                with open(tmp, "w") as f:
                    json.dump(state, f, indent=2, default=str)
                tmp.replace(self.json_state_file)

        except Exception as e:
            self.logger.debug(f"Could not save state for oversight: {e}")

    @staticmethod
    def _file_has_data(path) -> bool:
        """True if ``path`` exists and unpickles to a non-empty container."""
        try:
            if not Path(path).exists():
                return False
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return bool(obj)
        except Exception:
            # Unreadable/corrupt -> treat as no data so we can overwrite it.
            return False

    @staticmethod
    def _atomic_pickle(path, obj) -> None:
        """Pickle ``obj`` to ``path`` atomically (write tmp, then replace)."""
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        tmp.replace(path)
