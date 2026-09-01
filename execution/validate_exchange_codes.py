#!/usr/bin/env python3
"""
Pre-Live Exchange Code Validator

Validates and caches exchange codes for international stocks before live trading.
Can run with or without IBKR connection.

Usage:
    # Static validation only (no IBKR needed)
    python validate_exchange_codes.py

    # With IBKR connection for live contract details
    python validate_exchange_codes.py --live

    # Custom symbols
    python validate_exchange_codes.py --symbols CEZ.F 8002.T NVDA III.L

    # From config
    python validate_exchange_codes.py --from-config
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from exchange_manager import ExchangeManager


CACHE_FILE = Path(__file__).parent / 'exchange_codes_cache.json'


def load_cache() -> Dict:
    """Load existing cache"""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: Dict):
    """Save cache to file"""
    cache['_metadata'] = {
        'last_updated': datetime.now().isoformat(),
        'total_symbols': len([k for k in cache.keys() if not k.startswith('_')])
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"\n✅ Cache saved to: {CACHE_FILE}")


def validate_symbols_static(symbols: List[str], em: ExchangeManager) -> Dict:
    """
    Validate symbols using static mapping (no IBKR connection needed)

    Returns:
        Dict with validation results per symbol
    """
    results = {}

    print("\n" + "="*70)
    print("EXCHANGE CODE VALIDATION (Static)")
    print("="*70)
    print(f"{'Symbol':<18} {'IBKR':<12} {'Exchange':<8} {'Currency':<6} {'Lot':<6} {'Status'}")
    print("-"*70)

    errors = []
    warnings = []

    for symbol in symbols:
        try:
            ibkr_symbol, exchange, currency = em.parse_symbol(symbol)
            lot_size = em.get_lot_size(symbol)

            result = {
                'yfinance_symbol': symbol,
                'ibkr_symbol': ibkr_symbol,
                'exchange': exchange,
                'currency': currency,
                'lot_size': lot_size,
                'valid': True,
                'validated_at': datetime.now().isoformat(),
                'source': 'static'
            }

            status = "✓"

            # Check for potential issues
            if '.' in symbol and exchange == 'SMART':
                result['warning'] = "Has suffix but mapped to SMART - verify mapping!"
                result['valid'] = False
                warnings.append(symbol)
                status = "⚠️  UNMAPPED"

            results[symbol] = result

            print(f"{symbol:<18} {ibkr_symbol:<12} {exchange:<8} {currency:<6} {lot_size:<6} {status}")

        except Exception as e:
            results[symbol] = {
                'yfinance_symbol': symbol,
                'valid': False,
                'error': str(e),
                'validated_at': datetime.now().isoformat()
            }
            errors.append(symbol)
            print(f"{symbol:<18} {'ERROR':<12} {'':<8} {'':<6} {'':<6} ✗ {e}")

    print("-"*70)

    valid_count = sum(1 for r in results.values() if r.get('valid', False))
    print(f"\nSummary: {valid_count}/{len(symbols)} valid")

    if errors:
        print(f"❌ Errors: {errors}")
    if warnings:
        print(f"⚠️  Warnings: {warnings}")

    return results


def validate_symbols_live(symbols: List[str], em: ExchangeManager, port: int = 4002) -> Dict:
    """
    Validate symbols with live IBKR connection to fetch contract details

    Returns:
        Dict with validation results including live contract details
    """
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    import threading
    import time

    class IBClient(EWrapper, EClient):
        def __init__(self):
            EClient.__init__(self, self)
            self.contract_details = {}
            self.pending = {}
            self.events = {}

        def contractDetails(self, reqId, contractDetails):
            symbol = self.pending.get(reqId)
            if symbol:
                self.contract_details[symbol] = {
                    'conId': contractDetails.contract.conId,
                    'longName': contractDetails.longName,
                    'priceMagnifier': contractDetails.priceMagnifier,
                    'minTick': contractDetails.minTick,
                    'exchange': contractDetails.contract.exchange,
                    'currency': contractDetails.contract.currency,
                    'primaryExchange': contractDetails.contract.primaryExchange,
                }

        def contractDetailsEnd(self, reqId):
            if reqId in self.events:
                self.events[reqId].set()

        def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
            # Handle both old and new API signatures
            # New API: error(reqId, errorTime, errorCode, errorString, advancedOrderRejectJson)
            if reqId >= 0:
                symbol = self.pending.get(reqId)
                if symbol and errorCode not in [2104, 2106, 2158]:  # Ignore connection msgs
                    self.contract_details[symbol] = {'error': f"{errorCode}: {errorString}"}
                    if reqId in self.events:
                        self.events[reqId].set()

    # Connect to IBKR
    print("\n" + "="*70)
    print("EXCHANGE CODE VALIDATION (Live IBKR)")
    print("="*70)
    print(f"Connecting to IBKR Gateway on port {port}...")

    client = IBClient()
    client.connect('127.0.0.1', port, clientId=99)

    # Start message thread
    thread = threading.Thread(target=client.run, daemon=True)
    thread.start()
    time.sleep(1)

    if not client.isConnected():
        print("❌ Failed to connect to IBKR Gateway")
        print("   Make sure TWS/Gateway is running on port", port)
        return {}

    print("✓ Connected to IBKR")
    print()
    print(f"{'Symbol':<18} {'Exchange':<10} {'Currency':<6} {'ConId':<12} {'Status'}")
    print("-"*70)

    results = {}
    reqId = 1000

    for symbol in symbols:
        # Parse and create contract
        ibkr_symbol, exchange, currency = em.parse_symbol(symbol)

        contract = Contract()
        contract.symbol = ibkr_symbol
        contract.secType = 'STK'
        contract.exchange = exchange
        contract.currency = currency

        # Setup event
        client.events[reqId] = threading.Event()
        client.pending[reqId] = symbol

        # Request details
        client.reqContractDetails(reqId, contract)

        # Wait for response
        if client.events[reqId].wait(timeout=5):
            details = client.contract_details.get(symbol, {})

            if 'error' in details:
                results[symbol] = {
                    'yfinance_symbol': symbol,
                    'ibkr_symbol': ibkr_symbol,
                    'exchange': exchange,
                    'currency': currency,
                    'valid': False,
                    'error': details['error'],
                    'validated_at': datetime.now().isoformat(),
                    'source': 'live'
                }
                print(f"{symbol:<18} {exchange:<10} {currency:<6} {'':<12} ✗ {details['error']}")
            else:
                results[symbol] = {
                    'yfinance_symbol': symbol,
                    'ibkr_symbol': ibkr_symbol,
                    'exchange': details.get('exchange', exchange),
                    'currency': details.get('currency', currency),
                    'lot_size': em.get_lot_size(symbol),
                    'conId': details.get('conId'),
                    'longName': details.get('longName'),
                    'priceMagnifier': details.get('priceMagnifier', 1),
                    'minTick': details.get('minTick'),
                    'primaryExchange': details.get('primaryExchange'),
                    'valid': True,
                    'validated_at': datetime.now().isoformat(),
                    'source': 'live'
                }
                conId = details.get('conId', 'N/A')
                print(f"{symbol:<18} {exchange:<10} {currency:<6} {conId:<12} ✓")
        else:
            results[symbol] = {
                'yfinance_symbol': symbol,
                'ibkr_symbol': ibkr_symbol,
                'exchange': exchange,
                'currency': currency,
                'valid': False,
                'error': 'Timeout',
                'validated_at': datetime.now().isoformat(),
                'source': 'live'
            }
            print(f"{symbol:<18} {exchange:<10} {currency:<6} {'':<12} ✗ Timeout")

        reqId += 1
        time.sleep(0.5)  # Rate limit

    print("-"*70)

    # Disconnect
    client.disconnect()

    valid_count = sum(1 for r in results.values() if r.get('valid', False))
    print(f"\nSummary: {valid_count}/{len(symbols)} validated with IBKR")

    return results


def get_symbols_from_config() -> List[str]:
    """Get symbols from config.py"""
    try:
        from config import SYMBOLS
        return SYMBOLS
    except ImportError:
        print("Warning: Could not import SYMBOLS from config.py")
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Validate and cache exchange codes for international stocks'
    )
    parser.add_argument('--symbols', nargs='+', help='Symbols to validate')
    parser.add_argument('--from-config', action='store_true', help='Use SYMBOLS from config.py')
    parser.add_argument('--live', action='store_true', help='Connect to IBKR for live validation')
    parser.add_argument('--port', type=int, default=4002, help='IBKR Gateway port (default: 4002)')
    parser.add_argument('--show-cache', action='store_true', help='Show cached codes and exit')

    args = parser.parse_args()

    # Show cache only
    if args.show_cache:
        cache = load_cache()
        if cache:
            print("\nCached Exchange Codes:")
            print("="*70)
            for symbol, data in cache.items():
                if symbol.startswith('_'):
                    continue
                valid = "✓" if data.get('valid') else "✗"
                exchange = data.get('exchange', 'N/A')
                currency = data.get('currency', 'N/A')
                source = data.get('source', 'unknown')
                print(f"{symbol:<18} {exchange:<10} {currency:<6} [{source}] {valid}")
            print("="*70)
            meta = cache.get('_metadata', {})
            print(f"Last updated: {meta.get('last_updated', 'N/A')}")
            print(f"Total symbols: {meta.get('total_symbols', 0)}")
        else:
            print("No cache found. Run validation first.")
        return

    # Determine symbols to validate
    if args.symbols:
        symbols = args.symbols
    elif args.from_config:
        symbols = get_symbols_from_config()
        if not symbols:
            print("No symbols found in config.py")
            sys.exit(1)
    else:
        # Default test symbols
        symbols = [
            'NVDA',           # US
            '8002.T',         # Tokyo
            'III.L',          # London
            'CEZ.F',          # Frankfurt
            'BMW.DE',         # Germany XETRA
            '1919.HK',        # Hong Kong
            'BHP.AX',         # Australia
            'TATASTEEL.NS',   # India NSE
        ]
        print("Using default test symbols. Use --symbols or --from-config for custom list.")

    # Initialize exchange manager
    em = ExchangeManager()

    # Load existing cache
    cache = load_cache()

    # Validate
    if args.live:
        results = validate_symbols_live(symbols, em, args.port)
    else:
        results = validate_symbols_static(symbols, em)

    # Update cache
    cache.update(results)
    save_cache(cache)

    # Exit code based on validation
    all_valid = all(r.get('valid', False) for r in results.values())
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
