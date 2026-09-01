#!/usr/bin/env python3
"""
Test symbol parsing and conversion for all supported exchanges.

This script validates that the ExchangeManager correctly converts between
yfinance format and IBKR API format for all supported exchanges.

Usage:
    python test_symbol_parsing.py
"""

import sys
import logging
from exchange_manager import ExchangeManager


def setup_test_logger():
    """Setup a simple logger for test output"""
    logger = logging.getLogger('SymbolParsingTest')
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main():
    """Run symbol parsing tests"""
    print("="*60)
    print("SYMBOL PARSING & CONVERSION TEST")
    print("="*60)
    print()

    # Create exchange manager with logger
    logger = setup_test_logger()
    exchange_mgr = ExchangeManager(logger)

    print("Testing symbol conversions for all supported exchanges...")
    print()

    # Run tests
    results = exchange_mgr.test_symbol_conversions()

    # Count results
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    total = len(results)

    # Print summary
    print()
    print("="*60)
    print(f"TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("="*60)

    if failed > 0:
        print()
        print("❌ Some tests failed. See output above for details.")
        print()

        # List failed tests
        print("Failed tests:")
        for test_name, result in results.items():
            if not result:
                print(f"  - {test_name}")

        sys.exit(1)
    else:
        print()
        print("✅ All symbol parsing tests passed!")
        print()
        print("Supported exchanges:")
        print("  - NASDAQ (US): NVDA -> NVDA on SMART (USD)")
        print("  - Tokyo (Japan): 8002.T -> 8002 on TSEJ (JPY)")
        print("  - London (UK): III.L -> III on LSE (GBP)")
        print("  - NSE (India): TATASTEEL.NS -> TATASTEEL on NSE (INR)")
        print("  - BSE (India): TATASTEEL.BO -> TATASTEEL on BSE (INR)")
        print("  - Hong Kong: 0700.HK -> 0700 on SEHK (HKD)")
        print("  - Australia: BHP.AX -> BHP on ASX (AUD)")
        print("  - Paris: AIR.PA -> AIR on SBF (EUR)")
        print("  - Germany: BMW.DE -> BMW on IBIS (EUR)")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
