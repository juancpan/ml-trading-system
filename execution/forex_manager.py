"""
Forex Manager for JPY Carry Trade Strategy

After stock trades are executed, converts all non-JPY debt to JPY debt
by buying forex pairs. This creates a JPY carry trade position.

Two conversion methods:
1. Direct conversion: XXX → JPY (for USD, GBP, EUR, AUD, HKD, SGD, CAD, CHF, NZD)
2. Two-leg conversion: XXX → USD → JPY (for exotic currencies without direct JPY pairs)

Example (Direct):
- Account has -$30,000 USD debt and -£10,000 GBP debt
- Forex manager buys $30,000 worth of USD.JPY and £10,000 worth of GBP.JPY
- Result: USD balance → $0, GBP balance → £0, JPY balance → more negative

Example (Two-leg for HUF):
- Account has -100,000 HUF debt from Budapest-listed stocks (e.g. XYZ.BD)
- Step 1: SELL ~$263 USD.HUF to cover HUF debt (creates USD debt)
- Step 2: BUY $263 USD.JPY to convert USD debt to JPY
- Result: HUF balance → 0, USD balance → 0, JPY balance → more negative

Supported Direct Currencies: USD, GBP, EUR, AUD, HKD, SGD, CAD, CHF, NZD
Supported Two-Leg Currencies: HUF, SEK, CZK, NOK, PLN, DKK, ILS, MXN, ZAR, TRY, RON, INR
Unsupported (no liquid forex): SAR (Saudi Riyal - pegged to USD at 3.75)
"""

from typing import Dict, List, Optional, Tuple, Set
from ibapi.contract import Contract
from ibapi.order import Order
import logging
import time
import random
from config import ENABLE_JPY_CARRY_TRADE, JPY_CARRY_TRADE_MIN_DEBT


class ForexManager:
    """
    Manages forex operations for JPY carry trade strategy.

    Intelligently detects currencies from portfolio config and existing positions,
    then converts all non-JPY debt to JPY via IDEALPRO forex pairs.
    """

    # Supported currency pairs for carry trade via IDEALPRO
    # All major currencies that can be converted to JPY directly
    JPY_PAIRS = {
        'USD': 'USD.JPY',
        'GBP': 'GBP.JPY',
        'EUR': 'EUR.JPY',
        'AUD': 'AUD.JPY',
        'HKD': 'HKD.JPY',
        'SGD': 'SGD.JPY',  # Singapore Dollar
        'CAD': 'CAD.JPY',  # Canadian Dollar
        'CHF': 'CHF.JPY',  # Swiss Franc
        'NZD': 'NZD.JPY',  # New Zealand Dollar
    }

    # Two-leg conversion pairs for exotic currencies without direct JPY pairs
    # These currencies are converted via USD: XXX → USD → JPY
    # Format: currency -> (pair, action, is_quote_currency)
    # - pair: The USD/XXX pair available on IDEALPRO
    # - action: 'SELL' means sell USD to buy XXX (covers XXX debt, creates USD debt)
    # - is_quote_currency: True if XXX is the quote currency (USD.XXX), False if base (XXX.USD)
    TWO_LEG_PAIRS = {
        'HUF': ('USD.HUF', 'SELL', True),   # Hungarian Forint - Budapest-listed tickers
        'SEK': ('USD.SEK', 'SELL', True),   # Swedish Krona - SAAB-B.ST
        'CZK': ('USD.CZK', 'SELL', True),   # Czech Koruna - MONET.PR
        'NOK': ('USD.NOK', 'SELL', True),   # Norwegian Krone - EQNR.OL
        'PLN': ('USD.PLN', 'SELL', True),   # Polish Zloty - PKO.WA
        'DKK': ('USD.DKK', 'SELL', True),   # Danish Krone - NOVO-B.CO
        'ILS': ('USD.ILS', 'SELL', True),   # Israeli Shekel - TEVA.TA
        'MXN': ('USD.MXN', 'SELL', True),   # Mexican Peso
        'ZAR': ('USD.ZAR', 'SELL', True),   # South African Rand
        'TRY': ('USD.TRY', 'SELL', True),   # Turkish Lira
        'RON': ('USD.RON', 'SELL', True),   # Romanian Leu - TLV.RO
        'INR': ('USD.INR', 'SELL', True),   # Indian Rupee - TATASTEEL.NS (if JPY pair unavailable)
        # Note: SAR (Saudi Riyal) is pegged to USD at 3.75, may not have liquid forex pair
    }

    # Approximate exchange rates for size estimation (updated manually or via API)
    # These are fallback values - actual trades use market prices
    APPROX_RATES = {
        'HUF': 380,   # 1 USD = 380 HUF
        'SEK': 10.5,  # 1 USD = 10.5 SEK
        'CZK': 23.5,  # 1 USD = 23.5 CZK
        'NOK': 10.8,  # 1 USD = 10.8 NOK
        'PLN': 4.0,   # 1 USD = 4.0 PLN
        'DKK': 6.9,   # 1 USD = 6.9 DKK
        'ILS': 3.7,   # 1 USD = 3.7 ILS
        'MXN': 17.0,  # 1 USD = 17 MXN
        'ZAR': 18.5,  # 1 USD = 18.5 ZAR
        'TRY': 32.0,  # 1 USD = 32 TRY
        'RON': 4.6,   # 1 USD = 4.6 RON
        'INR': 83.0,  # 1 USD = 83 INR
        'SAR': 3.75,  # 1 USD = 3.75 SAR (pegged)
    }

    # Exchange to currency mapping (for auto-detection from config)
    EXCHANGE_CURRENCY_MAP = {
        'SMART': 'USD',     # US stocks (NASDAQ, NYSE)
        'TSEJ': 'JPY',      # Tokyo Stock Exchange
        'LSE': 'GBP',       # London Stock Exchange
        'SEHK': 'HKD',      # Hong Kong Stock Exchange
        'ASX': 'AUD',       # Australian Securities Exchange
        'SGX': 'SGD',       # Singapore Exchange
        'SBF': 'EUR',       # Euronext Paris
        'IBIS': 'EUR',      # Deutsche Börse (XETRA)
        'NSE': 'INR',       # India National Stock Exchange
        'BSE': 'INR',       # India Bombay Stock Exchange
        'BUX': 'HUF',       # Budapest Stock Exchange
        'SFB': 'SEK',       # Stockholm Stock Exchange
        'VSE': 'CZK',       # Prague Stock Exchange
        'OSE': 'NOK',       # Oslo Stock Exchange
        'WSE': 'PLN',       # Warsaw Stock Exchange
        'CPH': 'DKK',       # Copenhagen Stock Exchange
        'TASE': 'ILS',      # Tel Aviv Stock Exchange
        'TADAWUL': 'SAR',   # Saudi Stock Exchange (Tadawul)
        'BVB': 'RON',       # Bucharest Stock Exchange
        'BMV': 'MXN',       # Mexico Stock Exchange
    }

    # Symbol suffix to currency mapping (for yfinance symbols)
    SUFFIX_CURRENCY_MAP = {
        '.T': 'JPY',        # Tokyo: 8002.T
        '.L': 'GBP',        # London: III.L
        '.HK': 'HKD',       # Hong Kong: 1919.HK
        '.AX': 'AUD',       # Australia: SLX.AX
        '.SI': 'SGD',       # Singapore: D05.SI
        '.PA': 'EUR',       # Paris: AIR.PA
        '.DE': 'EUR',       # Germany: BMW.DE
        '.F': 'EUR',        # Frankfurt: CEZ.F
        '.NS': 'INR',       # India NSE: TATASTEEL.NS
        '.BO': 'INR',       # India BSE: RELIANCE.BO
        '.BD': 'HUF',       # Budapest-listed tickers (suffix .BD)
        '.ST': 'SEK',       # Stockholm: SAAB-B.ST
        '.PR': 'CZK',       # Prague: MONET.PR
        '.OL': 'NOK',       # Oslo: EQNR.OL
        '.WA': 'PLN',       # Warsaw: PKO.WA
        '.CO': 'DKK',       # Copenhagen: NOVO-B.CO
        '.TA': 'ILS',       # Tel Aviv: TEVA.TA
        '.SR': 'SAR',       # Saudi (Tadawul): 1303.SR
        '.RO': 'RON',       # Bucharest: TLV.RO
        '.MX': 'MXN',       # Mexico: WALMEX.MX
    }

    def __init__(self, ib_client, logger: Optional[logging.Logger] = None, exchange_manager=None):
        """
        Initialize forex manager

        Args:
            ib_client: IBClient instance for executing trades
            logger: Logger instance
            exchange_manager: ExchangeManager for currency detection (optional)
        """
        self.ib = ib_client
        self.logger = logger or logging.getLogger(__name__)
        self.exchange_manager = exchange_manager
        self.enabled = ENABLE_JPY_CARRY_TRADE
        self.min_debt_threshold = JPY_CARRY_TRADE_MIN_DEBT

    def get_expected_currencies_from_config(self) -> Set[str]:
        """
        Detect all currencies that SHOULD exist based on portfolio config.
        This is critical for requesting correct currency balances BEFORE
        positions are established.

        Returns:
            Set of currency codes expected from config (e.g., {'USD', 'JPY', 'HKD'})
        """
        from config import SYMBOLS

        currencies = {'USD'}  # Always include USD

        for symbol in SYMBOLS:
            # Method 1: Use exchange_manager if available
            if self.exchange_manager:
                currency = self.exchange_manager.get_currency(symbol)
                if currency:
                    currencies.add(currency)
                    continue

            # Method 2: Detect from symbol suffix
            for suffix, currency in self.SUFFIX_CURRENCY_MAP.items():
                if symbol.endswith(suffix):
                    currencies.add(currency)
                    break
            else:
                # No suffix = US stock (USD)
                if '.' not in symbol:
                    currencies.add('USD')

        self.logger.info(f"Expected currencies from config: {sorted(currencies)}")
        return currencies

    def analyze_currency_positions(self, account_values: Dict) -> Dict[str, float]:
        """
        Analyze currency positions from account values.
        Uses TotalCashBalance to avoid double-counting (prevents $215K error from summing duplicates).

        Args:
            account_values: Account values dict from IBKR

        Returns:
            Dictionary of {currency: balance} for all currencies with balances
        """
        currency_balances = {}

        # PREFERRED: Use per-currency balances from reqAccountSummary (ib_client)
        # This is THE ONLY way to detect GBP, JPY, EUR individual debts
        if hasattr(self.ib, 'currency_balances') and self.ib.currency_balances:
            self.logger.info(f"Using per-currency balances from reqAccountSummary:")
            for curr, bal in self.ib.currency_balances.items():
                self.logger.info(f"  {curr}: {bal:.2f}")
            return dict(self.ib.currency_balances)

        # FALLBACK: Parse from account_values (LIMITED - USD only!)
        # reqAccountUpdates() only returns USD equivalent - cannot detect GBP/JPY debts
        self.logger.warning("="*60)
        self.logger.warning("Currency balances NOT available from reqAccountSummary!")
        self.logger.warning("Will only detect USD - GBP/JPY debts will be MISSED!")
        self.logger.warning("This means -£31,732 GBP debt will NOT be converted to JPY!")
        self.logger.warning("="*60)

        # Try to get USD balance only
        if 'TotalCashBalance' in account_values:
            data = account_values['TotalCashBalance']
            currency = data.get('currency', 'USD')
            balance = float(data.get('value', 0))

            if currency in ['USD', 'BASE']:
                currency_balances['USD'] = balance
                self.logger.info(f"Fallback USD balance: {balance:.2f} USD")

        return currency_balances

    def identify_carry_trade_opportunities(self, currency_balances: Dict[str, float]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, str, float, float]]]:
        """
        Identify currencies with negative balances (debt) that can be converted to JPY

        Args:
            currency_balances: Dictionary of {currency: balance}

        Returns:
            Tuple of:
            - direct_opportunities: List of (currency_pair, amount_to_trade) for direct JPY pairs
            - two_leg_opportunities: List of (currency, intermediate_pair, debt_amount, usd_equivalent)
              for currencies requiring two-leg conversion via USD
        """
        direct_opportunities = []
        two_leg_opportunities = []

        for currency, balance in currency_balances.items():
            # Skip JPY itself
            if currency == 'JPY':
                continue

            # Only process negative balances (debt)
            if balance >= 0:
                continue

            debt_amount = abs(balance)

            # Check if we have a direct JPY pair for this currency
            if currency in self.JPY_PAIRS:
                # Check minimum debt threshold
                if debt_amount < self.min_debt_threshold:
                    self.logger.info(f"Skipping {currency} debt of {debt_amount:.2f} (below {self.min_debt_threshold} threshold)")
                    continue

                # Add to direct opportunities
                pair = self.JPY_PAIRS[currency]
                direct_opportunities.append((pair, debt_amount))
                self.logger.info(f"Found direct carry trade: {pair} for {debt_amount:.2f} {currency}")

            elif currency in self.TWO_LEG_PAIRS:
                # Two-leg conversion: XXX → USD → JPY
                intermediate_pair, action, is_quote = self.TWO_LEG_PAIRS[currency]

                # Estimate USD equivalent for threshold check
                approx_rate = self.APPROX_RATES.get(currency, 1.0)
                usd_equivalent = debt_amount / approx_rate

                if usd_equivalent < self.min_debt_threshold:
                    self.logger.info(f"Skipping {currency} debt of {debt_amount:.2f} (~${usd_equivalent:.2f} USD, below threshold)")
                    continue

                two_leg_opportunities.append((currency, intermediate_pair, debt_amount, usd_equivalent))
                self.logger.info(f"Found two-leg carry trade: {currency} ({debt_amount:.2f}) → USD (~${usd_equivalent:.2f}) → JPY")

            else:
                # No conversion path available
                self.logger.warning(f"Currency conversion not supported for {currency} - no JPY or USD pair configured")
                self.logger.warning(f"  Debt of {debt_amount:.2f} {currency} will NOT be converted to JPY")

        return direct_opportunities, two_leg_opportunities

    def create_forex_contract(self, pair: str) -> Contract:
        """
        Create IBKR Forex contract using NATIVE ibapi.

        Official IBKR API Format for Forex/Cash:
        - secType: 'CASH'
        - symbol: base currency
        - currency: quote currency
        - exchange: 'IDEALPRO'

        Args:
            pair: Currency pair in format 'USD.JPY'

        Returns:
            ibapi.contract.Contract configured for forex
        """
        # Split pair (e.g., 'USD.JPY' -> 'USD', 'JPY')
        base, quote = pair.split('.')

        # Create native API forex contract
        contract = Contract()
        contract.symbol = base
        contract.secType = 'CASH'
        contract.currency = quote
        contract.exchange = 'IDEALPRO'

        self.logger.debug(f"Created forex contract: {base}/{quote}")
        return contract

    def execute_carry_trades(self, direct_opportunities: List[Tuple[str, float]],
                             two_leg_opportunities: List[Tuple[str, str, float, float]] = None) -> Dict:
        """
        Execute carry trade orders (both direct and two-leg)

        Args:
            direct_opportunities: List of (pair, amount) tuples for direct XXX.JPY trades
            two_leg_opportunities: List of (currency, intermediate_pair, debt_amount, usd_equivalent)
                                   for two-leg conversions via USD

        Returns:
            Dictionary with execution results
        """
        two_leg_opportunities = two_leg_opportunities or []
        total_direct = len(direct_opportunities)
        total_two_leg = len(two_leg_opportunities)

        results = {
            'executed': [],
            'failed': [],
            'two_leg_executed': [],
            'two_leg_failed': [],
            'total_direct_trades': total_direct,
            'total_two_leg_trades': total_two_leg,
            'accumulated_usd_debt': 0.0  # Track USD debt from two-leg conversions
        }

        if not direct_opportunities and not two_leg_opportunities:
            self.logger.info("No carry trade opportunities to execute")
            return results

        # === STEP 1: Execute two-leg conversions first (XXX → USD) ===
        if two_leg_opportunities:
            self.logger.info(f"\n{'='*50}")
            self.logger.info("PHASE 1: Two-Leg Conversions (XXX → USD)")
            self.logger.info(f"{'='*50}")

            for currency, intermediate_pair, debt_amount, usd_equivalent in two_leg_opportunities:
                try:
                    # Create intermediate forex contract (e.g., USD.HUF)
                    contract = self.create_forex_contract(intermediate_pair)

                    # SELL USD.XXX means: sell USD, buy XXX (covers XXX debt, creates USD debt)
                    # Amount is in XXX (quote currency), but order quantity is in USD (base)
                    # So we trade the estimated USD equivalent
                    order = Order()
                    order.action = 'SELL'  # Sell USD to buy the exotic currency
                    order.totalQuantity = int(usd_equivalent)  # Amount in USD
                    order.orderType = "MKT"
                    order.tif = "DAY"
                    order.transmit = True

                    # Get next order ID
                    order_id = self.ib.nextValidOrderId
                    self.ib.nextValidOrderId += 1

                    # Place order
                    self.ib.placeOrder(order_id, contract, order)

                    # Add delay
                    delay = random.uniform(2.0, 3.0)
                    self.logger.debug(f"Waiting {delay:.2f}s after two-leg order for {intermediate_pair}")
                    time.sleep(delay)

                    self.logger.info(f"✅ Two-leg leg 1: SELL ${usd_equivalent:.0f} {intermediate_pair} to cover {debt_amount:.0f} {currency} debt (Order ID: {order_id})")

                    results['two_leg_executed'].append({
                        'currency': currency,
                        'intermediate_pair': intermediate_pair,
                        'debt_amount': debt_amount,
                        'usd_equivalent': usd_equivalent,
                        'order_id': order_id
                    })
                    results['accumulated_usd_debt'] += usd_equivalent

                except Exception as e:
                    self.logger.error(f"Failed two-leg conversion for {currency} via {intermediate_pair}: {e}")
                    results['two_leg_failed'].append({
                        'currency': currency,
                        'intermediate_pair': intermediate_pair,
                        'reason': str(e)
                    })

        # === STEP 2: Execute direct JPY conversions ===
        if direct_opportunities or results['accumulated_usd_debt'] > 0:
            self.logger.info(f"\n{'='*50}")
            self.logger.info("PHASE 2: Direct JPY Conversions")
            self.logger.info(f"{'='*50}")

        # Add accumulated USD debt from two-leg conversions to USD.JPY trade
        if results['accumulated_usd_debt'] > self.min_debt_threshold:
            # Check if there's already a USD.JPY trade in direct_opportunities
            usd_jpy_exists = any(pair == 'USD.JPY' for pair, _ in direct_opportunities)

            if usd_jpy_exists:
                # Add to existing USD.JPY trade
                direct_opportunities = [
                    (pair, amount + results['accumulated_usd_debt']) if pair == 'USD.JPY' else (pair, amount)
                    for pair, amount in direct_opportunities
                ]
                self.logger.info(f"Added ${results['accumulated_usd_debt']:.0f} from two-leg conversions to existing USD.JPY trade")
            else:
                # Create new USD.JPY trade for accumulated debt
                direct_opportunities.append(('USD.JPY', results['accumulated_usd_debt']))
                self.logger.info(f"Created USD.JPY trade for ${results['accumulated_usd_debt']:.0f} from two-leg conversions")

        # Execute direct trades
        for pair, amount in direct_opportunities:
            try:
                # Create forex contract
                contract = self.create_forex_contract(pair)

                # BUY XXX.JPY means: buy XXX, sell JPY (converts XXX debt to JPY debt)
                order = Order()
                order.action = 'BUY'
                order.totalQuantity = int(amount)  # Forex quantity in base currency units
                order.orderType = "MKT"
                order.tif = "DAY"
                order.transmit = True

                # Get next order ID
                order_id = self.ib.nextValidOrderId
                self.ib.nextValidOrderId += 1

                # Place order
                self.ib.placeOrder(order_id, contract, order)

                # Add delay
                delay = random.uniform(2.0, 3.0)
                self.logger.debug(f"Waiting {delay:.2f}s after carry trade order for {pair}")
                time.sleep(delay)

                self.logger.info(f"✅ Direct carry trade: BUY {amount:.0f} {pair} (Order ID: {order_id})")
                results['executed'].append({
                    'pair': pair,
                    'amount': amount,
                    'order_id': order_id
                })

            except Exception as e:
                self.logger.error(f"Failed to execute carry trade for {pair}: {e}")
                results['failed'].append({'pair': pair, 'reason': str(e)})

        return results

    def run_carry_trade_strategy(self, account_values: Dict, force_currencies: List[str] = None) -> Dict:
        """
        Main entry point: Run complete carry trade strategy

        Args:
            account_values: Account values dict from IBKR
            force_currencies: Optional list of currencies to explicitly check
                             (useful when positions not yet established)

        Returns:
            Dictionary with execution results
        """
        if not self.enabled:
            self.logger.info("JPY carry trade is disabled in config")
            return {'enabled': False}

        self.logger.info("="*60)
        self.logger.info("JPY CARRY TRADE STRATEGY")
        self.logger.info("="*60)

        # Step 0: Get expected currencies from config
        expected_currencies = self.get_expected_currencies_from_config()
        self.logger.info(f"Portfolio currencies: {sorted(expected_currencies)}")

        # If force_currencies provided, ensure they're requested
        if force_currencies:
            for curr in force_currencies:
                expected_currencies.add(curr)
            self.logger.info(f"Including forced currencies: {force_currencies}")

        # Step 1: Analyze currency positions
        currency_balances = self.analyze_currency_positions(account_values)

        if not currency_balances:
            self.logger.warning("No currency balances found in account")
            self.logger.warning("This may indicate ib_client.request_currency_balances() failed")
            return {'enabled': True, 'executed': [], 'failed': [], 'balances': {}}

        # Display currency balance summary table
        self._log_currency_summary(currency_balances, expected_currencies)

        # Step 2: Identify carry trade opportunities (direct and two-leg)
        direct_opportunities, two_leg_opportunities = self.identify_carry_trade_opportunities(currency_balances)

        if not direct_opportunities and not two_leg_opportunities:
            self.logger.info("No carry trade opportunities identified")
            self.logger.info("  (All non-JPY currencies have positive or zero balances)")
            return {'enabled': True, 'executed': [], 'failed': [], 'two_leg_executed': [], 'two_leg_failed': [], 'balances': currency_balances}

        # Step 3: Execute trades (both direct and two-leg)
        results = self.execute_carry_trades(direct_opportunities, two_leg_opportunities)
        results['enabled'] = True
        results['balances'] = currency_balances

        # Log summary
        self.logger.info("="*60)
        self.logger.info("CARRY TRADE SUMMARY")
        self.logger.info(f"Direct opportunities: {results['total_direct_trades']}")
        self.logger.info(f"Two-leg opportunities: {results['total_two_leg_trades']}")
        self.logger.info(f"Direct executed: {len(results['executed'])}")
        self.logger.info(f"Two-leg executed: {len(results.get('two_leg_executed', []))}")
        self.logger.info(f"Direct failed: {len(results['failed'])}")
        self.logger.info(f"Two-leg failed: {len(results.get('two_leg_failed', []))}")

        # Show what was converted (direct)
        if results['executed']:
            self.logger.info("\nDirect conversions executed (XXX → JPY):")
            for trade in results['executed']:
                pair = trade['pair']
                amount = trade['amount']
                base_curr = pair.split('.')[0]
                self.logger.info(f"  {base_curr}: {amount:,.2f} → JPY (via {pair})")

        # Show what was converted (two-leg)
        if results.get('two_leg_executed'):
            self.logger.info("\nTwo-leg conversions executed (XXX → USD → JPY):")
            for trade in results['two_leg_executed']:
                currency = trade['currency']
                debt = trade['debt_amount']
                usd_eq = trade['usd_equivalent']
                self.logger.info(f"  {currency}: {debt:,.0f} → ~${usd_eq:,.0f} USD → JPY")

        if results['failed']:
            self.logger.warning("\nDirect conversions failed:")
            for fail in results['failed']:
                self.logger.warning(f"  {fail['pair']}: {fail['reason']}")

        if results.get('two_leg_failed'):
            self.logger.warning("\nTwo-leg conversions failed:")
            for fail in results['two_leg_failed']:
                self.logger.warning(f"  {fail['currency']} via {fail['intermediate_pair']}: {fail['reason']}")

        self.logger.info("="*60)

        return results

    def _log_currency_summary(self, balances: Dict[str, float], expected: Set[str]):
        """
        Log a summary table of currency balances with carry trade status.
        """
        self.logger.info("\n" + "-"*60)
        self.logger.info("CURRENCY BALANCE SUMMARY")
        self.logger.info("-"*60)
        self.logger.info(f"{'Currency':<10} {'Balance':>15} {'Status':<20} {'Action':<15}")
        self.logger.info("-"*60)

        # Process all currencies (both detected and expected)
        all_currencies = set(balances.keys()) | expected

        for currency in sorted(all_currencies):
            balance = balances.get(currency, None)

            if balance is None:
                status = "NOT DETECTED"
                action = "Check request"
            elif currency == 'JPY':
                status = "TARGET CURRENCY"
                action = "No action"
            elif balance < 0:
                debt = abs(balance)
                if debt >= self.min_debt_threshold:
                    status = f"DEBT ({debt:,.0f})"
                    if currency in self.JPY_PAIRS:
                        action = f"→ {self.JPY_PAIRS[currency]}"
                    elif currency in self.TWO_LEG_PAIRS:
                        pair, _, _ = self.TWO_LEG_PAIRS[currency]
                        action = f"→ {pair} → JPY"
                    else:
                        action = "NO PAIR!"
                else:
                    status = f"DEBT (small)"
                    action = f"< {self.min_debt_threshold} min"
            elif balance > 0:
                status = f"CREDIT ({balance:,.0f})"
                action = "No action"
            else:
                status = "ZERO"
                action = "No action"

            balance_str = f"{balance:>15,.2f}" if balance is not None else "        N/A"
            self.logger.info(f"{currency:<10} {balance_str} {status:<20} {action:<15}")

        self.logger.info("-"*60)

        # Warn about missing currencies that should have been detected
        missing = expected - set(balances.keys())
        if missing:
            self.logger.warning(f"⚠️  Missing currency balances (expected from config): {sorted(missing)}")
            self.logger.warning("   Possible causes:")
            self.logger.warning("   1. Positions not yet established (first run)")
            self.logger.warning("   2. ib_client.request_currency_balances() needs explicit currency list")
            self.logger.warning("   3. IBKR API timeout or error")


# Convenience function
def create_forex_manager(ib_client, logger: Optional[logging.Logger] = None) -> ForexManager:
    """Create and return a ForexManager instance"""
    return ForexManager(ib_client, logger)
