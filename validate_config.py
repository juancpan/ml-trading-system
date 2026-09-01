#!/usr/bin/env python3
"""
Configuration Validation Script

Validates all configuration files for the trading system:
- IBKR configuration (config.py)
- Crypto configuration (config.yaml)
- Model deployment manifests
- Exchange and leverage settings

Usage:
    python validate_config.py
    python validate_config.py --verbose
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

# Add paths
sys.path.append(str(Path(__file__).parent / "execution"))
sys.path.append(str(Path(__file__).parent / "crypto_trading"))


class ConfigValidator:
    """Validates trading system configuration"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.info = []

    def log(self, level: str, message: str):
        """Log validation message"""
        if level == "ERROR":
            self.errors.append(message)
            print(f"❌ ERROR: {message}")
        elif level == "WARNING":
            self.warnings.append(message)
            print(f"⚠️  WARNING: {message}")
        elif level == "INFO":
            self.info.append(message)
            if self.verbose:
                print(f"ℹ️  INFO: {message}")

    def validate_ibkr_config(self) -> bool:
        """Validate IBKR configuration"""
        print("\n" + "=" * 60)
        print("VALIDATING IBKR CONFIGURATION")
        print("=" * 60)

        try:
            # Import config
            import config

            # 1. Validate leverage mode
            if not hasattr(config, "LEVERAGE_MODE"):
                self.log("ERROR", "LEVERAGE_MODE not defined in config")
                return False

            if config.LEVERAGE_MODE not in ["portfolio_mode", "isolated_mode"]:
                self.log("ERROR", f"Invalid LEVERAGE_MODE: {config.LEVERAGE_MODE}")
                return False

            self.log("INFO", f"Leverage mode: {config.LEVERAGE_MODE}")

            # 2. Validate general leverage (portfolio mode)
            if config.LEVERAGE_MODE == "portfolio_mode":
                if not hasattr(config, "GENERAL_LEVERAGE"):
                    self.log("ERROR", "GENERAL_LEVERAGE not defined for portfolio_mode")
                    return False

                if config.GENERAL_LEVERAGE <= 0:
                    self.log(
                        "ERROR",
                        f"GENERAL_LEVERAGE must be positive: {config.GENERAL_LEVERAGE}",
                    )
                    return False

                if config.GENERAL_LEVERAGE > 5:
                    self.log(
                        "WARNING",
                        f"GENERAL_LEVERAGE is high: {config.GENERAL_LEVERAGE}x",
                    )

                self.log("INFO", f"General leverage: {config.GENERAL_LEVERAGE}x")

            # 3. Validate target allocation
            if not hasattr(config, "TARGET_ALLOCATION"):
                self.log("ERROR", "TARGET_ALLOCATION not defined")
                return False

            total_allocation = sum(config.TARGET_ALLOCATION.values())
            if abs(total_allocation - 1.0) > 0.01:
                self.log(
                    "WARNING",
                    f"TARGET_ALLOCATION sum: {total_allocation:.2%} (should be 100%)",
                )

            self.log(
                "INFO", f"Portfolio allocation: {len(config.TARGET_ALLOCATION)} assets"
            )

            # 4. Validate asset configs
            if not hasattr(config, "ASSET_SPECIFIC_CONFIGS"):
                self.log("ERROR", "ASSET_SPECIFIC_CONFIGS not defined")
                return False

            for symbol in config.TARGET_ALLOCATION.keys():
                if symbol not in config.ASSET_SPECIFIC_CONFIGS:
                    self.log("ERROR", f"Missing ASSET_SPECIFIC_CONFIGS for {symbol}")
                    return False

                asset_config = config.ASSET_SPECIFIC_CONFIGS[symbol]

                # Check strategy type
                strategy_type = asset_config.get(
                    "strategy_type", config.DEFAULT_STRATEGY_TYPE
                )
                if strategy_type not in config.VALID_STRATEGY_TYPES:
                    self.log(
                        "ERROR",
                        f"{symbol}: Invalid strategy_type '{strategy_type}'. Must be one of {config.VALID_STRATEGY_TYPES}",
                    )
                    return False

                # For ml_signal: must have model_type and strategy_model_path
                if strategy_type == "ml_signal":
                    # Check model type
                    if "model_type" not in asset_config:
                        self.log(
                            "ERROR",
                            f"{symbol}: ml_signal strategy requires 'model_type'",
                        )
                        return False

                    # Check model path
                    if "strategy_model_path" not in asset_config:
                        self.log(
                            "ERROR",
                            f"{symbol}: ml_signal strategy requires 'strategy_model_path'",
                        )
                        return False

                    # Check if model file exists
                    model_path = (
                        Path(__file__).parent
                        / "execution"
                        / asset_config["strategy_model_path"]
                    )
                    if not model_path.exists():
                        self.log(
                            "WARNING", f"{symbol}: model file not found: {model_path}"
                        )

                    self.log(
                        "INFO",
                        f"{symbol}: {asset_config['model_type']} model (ML signal strategy)",
                    )

                # For buy_and_hold: model fields optional
                elif strategy_type == "buy_and_hold":
                    self.log(
                        "INFO", f"{symbol}: Buy-and-hold strategy (no ML model needed)"
                    )

                # Check minimum position shares
                min_shares = asset_config.get("min_position_shares", None)
                if min_shares is not None:
                    # Validate it's a positive number
                    if not isinstance(min_shares, (int, float)) or min_shares < 0:
                        self.log(
                            "ERROR",
                            f"{symbol}: min_position_shares must be positive number or None (got: {min_shares})",
                        )
                        return False

                    if min_shares > 0:
                        # Check lot size compatibility
                        try:
                            lot_size = exchange_mgr.get_lot_size(symbol)
                            if lot_size > 1 and min_shares % lot_size != 0:
                                self.log(
                                    "WARNING",
                                    f"{symbol}: min_position_shares ({min_shares}) not multiple of lot size ({lot_size}) "
                                    f"for exchange {exchange_mgr.get_exchange(symbol)}",
                                )
                        except Exception as e:
                            pass  # Skip if exchange manager check fails

                        self.log(
                            "INFO", f"{symbol}: Minimum position = {min_shares} shares"
                        )

                # Check Kelly fraction (isolated mode)
                if config.LEVERAGE_MODE == "isolated_mode":
                    if "kelly_fraction" not in asset_config:
                        self.log(
                            "WARNING",
                            f"{symbol}: missing kelly_fraction for isolated_mode",
                        )

            # 5. Validate JPY carry trade config (legacy)
            if hasattr(config, "ENABLE_JPY_CARRY_TRADE"):
                self.log(
                    "INFO",
                    f"JPY carry trade (legacy): {'ENABLED' if config.ENABLE_JPY_CARRY_TRADE else 'DISABLED'}",
                )

                if config.ENABLE_JPY_CARRY_TRADE and hasattr(
                    config, "JPY_CARRY_TRADE_MIN_DEBT"
                ):
                    self.log(
                        "INFO",
                        f"Min debt threshold: ${config.JPY_CARRY_TRADE_MIN_DEBT}",
                    )

            # 5b. Validate Cash Portfolio Config (carry trade engine)
            cash_valid = self._validate_cash_portfolio_config(config)
            if not cash_valid:
                return False

            # 6. Validate symbols list
            if hasattr(config, "SYMBOLS"):
                if set(config.SYMBOLS) != set(config.TARGET_ALLOCATION.keys()):
                    self.log(
                        "WARNING", "SYMBOLS list doesn't match TARGET_ALLOCATION keys"
                    )

            # 7. Validate exchange support
            try:
                from execution.exchange_manager import ExchangeManager
                import logging

                test_logger = logging.getLogger("exchange_validation")
                test_logger.setLevel(logging.CRITICAL)  # Suppress logs
                exchange_mgr = ExchangeManager(test_logger)

                supported_exchanges = [
                    "TSEJ",
                    "LSE",
                    "SEHK",
                    "ASX",
                    "SGX",
                    "SBF",
                    "IBIS",
                    "FWB2",
                    "NSE",
                    "BSE",
                    "BM",
                    "SMART",
                ]

                for symbol in config.SYMBOLS:
                    try:
                        exchange = exchange_mgr.get_exchange(symbol)
                        if exchange not in supported_exchanges:
                            self.log(
                                "WARNING", f"{symbol}: Unknown exchange {exchange}"
                            )
                        else:
                            self.log(
                                "INFO", f"{symbol}: Validated for exchange {exchange}"
                            )
                    except Exception as e:
                        self.log(
                            "WARNING", f"{symbol}: Could not validate exchange: {e}"
                        )

            except Exception as e:
                self.log("WARNING", f"Could not validate exchanges: {e}")

            return len(self.errors) == 0

        except Exception as e:
            self.log("ERROR", f"Failed to import IBKR config: {e}")
            return False

    def _validate_cash_portfolio_config(self, config) -> bool:
        """Validate CASH_PORTFOLIO_CONFIG for the carry trade engine.

        This validates the cash portfolio configuration when CASH_REBALANCING_MODE
        is set to 'phase1' or 'phase2'. In 'legacy' mode, this config is ignored.

        For 'phase2' mode: every carry trade model file MUST exist on disk.
        If a model is missing, this function logs a FATAL error and returns False,
        which causes the program to exit.

        Returns:
            True if valid, False if fatal errors found.
        """
        # Check if feature flag exists
        mode = getattr(config, "CASH_REBALANCING_MODE", "legacy")
        valid_modes = ("legacy", "phase1", "phase2")
        if mode not in valid_modes:
            self.log(
                "ERROR",
                f"CASH_REBALANCING_MODE must be one of {valid_modes}, got: '{mode}'",
            )
            return False

        self.log("INFO", f"Cash rebalancing mode: {mode}")

        # In legacy mode, CASH_PORTFOLIO_CONFIG is optional
        if mode == "legacy":
            self.log("INFO", "Legacy mode -- CASH_PORTFOLIO_CONFIG not required")
            return True

        # For phase1/phase2, CASH_PORTFOLIO_CONFIG is mandatory
        if not hasattr(config, "CASH_PORTFOLIO_CONFIG"):
            self.log(
                "ERROR",
                "CASH_PORTFOLIO_CONFIG is required when CASH_REBALANCING_MODE is not 'legacy'",
            )
            return False

        cfg = config.CASH_PORTFOLIO_CONFIG

        if not isinstance(cfg, dict):
            self.log("ERROR", "CASH_PORTFOLIO_CONFIG must be a dictionary")
            return False

        # Validate required keys (single-pair carry trade: USD ↔ JPY)
        required_keys = [
            "funding_currency",
            "carry_currency",
            "carry_pair",
            "carry_model",
            "exotic_routing",
        ]
        for key in required_keys:
            if key not in cfg:
                self.log(
                    "ERROR", f"CASH_PORTFOLIO_CONFIG missing required key: '{key}'"
                )
                return False

        # Validate funding_currency and carry_currency
        funding_ccy = cfg["funding_currency"]
        carry_ccy = cfg["carry_currency"]
        if not isinstance(funding_ccy, str) or len(funding_ccy) != 3:
            self.log("ERROR", f"funding_currency must be a 3-letter string, got: '{funding_ccy}'")
            return False
        if not isinstance(carry_ccy, str) or len(carry_ccy) != 3:
            self.log("ERROR", f"carry_currency must be a 3-letter string, got: '{carry_ccy}'")
            return False
        if funding_ccy == carry_ccy:
            self.log("ERROR", f"funding_currency and carry_currency must differ (both are '{funding_ccy}')")
            return False

        # Validate carry_pair matches funding + carry
        carry_pair = cfg["carry_pair"]
        expected_pair = f"{funding_ccy}{carry_ccy}"
        if carry_pair != expected_pair:
            self.log(
                "ERROR",
                f"carry_pair '{carry_pair}' does not match "
                f"funding_currency + carry_currency = '{expected_pair}'",
            )
            return False

        self.log("INFO", f"Carry trade: {funding_ccy} ↔ {carry_ccy} (pair: {carry_pair})")

        # Validate exotic_routing: all values must route to funding_currency
        routing = cfg["exotic_routing"]
        if not isinstance(routing, dict):
            self.log("ERROR", "exotic_routing must be a dictionary")
            return False

        for exotic_ccy, target in routing.items():
            if target != funding_ccy:
                self.log(
                    "ERROR",
                    f"exotic_routing['{exotic_ccy}'] routes to '{target}' "
                    f"but all exotics must route to funding_currency '{funding_ccy}'",
                )
                return False
            if exotic_ccy == funding_ccy:
                self.log(
                    "WARNING",
                    f"'{exotic_ccy}' is both the funding_currency and in exotic_routing "
                    f"(redundant — will be ignored in Phase 1)",
                )
            if exotic_ccy == carry_ccy:
                self.log(
                    "WARNING",
                    f"'{exotic_ccy}' is the carry_currency and in exotic_routing "
                    f"(carry currency should not be routed)",
                )

        self.log("INFO", f"Exotic routing: {len(routing)} currencies → {funding_ccy}")

        # Validate carry_model
        model_cfg = cfg["carry_model"]
        if not isinstance(model_cfg, dict):
            self.log("ERROR", "carry_model must be a dictionary")
            return False

        if "model_type" not in model_cfg:
            self.log("ERROR", f"carry_model missing 'model_type'")
            return False

        if "strategy_model_path" not in model_cfg:
            self.log("ERROR", f"carry_model missing 'strategy_model_path'")
            return False

        # Validate carry model file
        ibkr_dir = Path(__file__).parent / "execution"
        model_path = ibkr_dir / model_cfg["strategy_model_path"]

        if mode == "phase2":
            if not model_path.exists():
                # Not fatal — missing model fallback returns signal=+1 (convert)
                self.log(
                    "WARNING",
                    f"Carry model not found: {model_path}. "
                    f"Fallback: signal=+1 (immediate conversion). "
                    f"Deploy the model for ML-timed carry trades.",
                )
            else:
                import time
                model_age_days = (time.time() - model_path.stat().st_mtime) / 86400
                if model_age_days > 90:
                    self.log(
                        "WARNING",
                        f"Carry model for {carry_pair} is {model_age_days:.0f} days old. "
                        f"Consider retraining.",
                    )
                self.log(
                    "INFO",
                    f"{carry_pair}: {model_cfg['model_type']} model OK "
                    f"({model_age_days:.0f} days old)",
                )
        else:
            if model_path.exists():
                self.log("INFO", f"{carry_pair}: {model_cfg['model_type']} model found")
            else:
                self.log("INFO", f"{carry_pair}: model not yet deployed (OK for {mode} mode)")

        # Validate max_hold_days
        max_hold = cfg.get("max_hold_days", 30)
        if not isinstance(max_hold, int) or max_hold < 1:
            self.log("ERROR", "max_hold_days must be a positive integer")
            return False

        self.log("INFO", f"Max hold duration: {max_hold} business days")

        # Validate forex_order_type
        order_type = cfg.get("forex_order_type", "MKT")
        if order_type not in ("LMT", "MKT"):
            self.log(
                "ERROR",
                f"forex_order_type must be 'LMT' or 'MKT', got: '{order_type}'. "
                f"Note: MIDPRICE is NOT supported on IDEALPRO.",
            )
            return False

        self.log("INFO", f"Forex order type: {order_type}")

        return True

    def validate_crypto_config(self) -> bool:
        """Validate crypto configuration"""
        print("\n" + "=" * 60)
        print("VALIDATING CRYPTO CONFIGURATION")
        print("=" * 60)

        try:
            # Load config
            config_path = Path(__file__).parent / "crypto_trading" / "config.yaml"
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # 1. Validate leverage mode
            leverage_mode = config.get("portfolio", {}).get(
                "leverage_mode", "isolated_mode"
            )
            if leverage_mode not in ["portfolio_mode", "isolated_mode"]:
                self.log("ERROR", f"Invalid leverage_mode: {leverage_mode}")
                return False

            self.log("INFO", f"Leverage mode: {leverage_mode}")

            # 2. Validate general leverage (portfolio mode)
            if leverage_mode == "portfolio_mode":
                general_leverage = config.get("portfolio", {}).get("general_leverage")
                if general_leverage is None:
                    self.log("ERROR", "general_leverage not defined for portfolio_mode")
                    return False

                if general_leverage <= 0:
                    self.log(
                        "ERROR",
                        f"general_leverage must be positive: {general_leverage}",
                    )
                    return False

                if general_leverage > 5:
                    self.log(
                        "WARNING", f"general_leverage is high: {general_leverage}x"
                    )

                self.log("INFO", f"General leverage: {general_leverage}x")

            # 3. Validate target allocation
            target_allocation = config.get("portfolio", {}).get("target_allocation", {})
            if not target_allocation:
                self.log("WARNING", "No target_allocation defined")
            else:
                # Check that all symbols in trading.symbols have allocations
                symbols = config.get("trading", {}).get("symbols", [])
                for symbol in symbols:
                    # Check various symbol formats
                    found = False
                    for key in target_allocation.keys():
                        if symbol in key or key in symbol:
                            found = True
                            break

                    if not found:
                        self.log(
                            "WARNING", f"No allocation for trading symbol: {symbol}"
                        )

                self.log(
                    "INFO", f"Portfolio allocation: {len(target_allocation)} entries"
                )

            # 4. Validate Kelly fractions (isolated mode)
            if leverage_mode == "isolated_mode":
                kelly_fractions = config.get("portfolio", {}).get(
                    "asset_kelly_fractions", {}
                )
                if not kelly_fractions:
                    self.log(
                        "WARNING", "No asset_kelly_fractions defined for isolated_mode"
                    )

            # 5. Validate exchange config
            exchange_name = config.get("exchange", {}).get("name")
            testnet = config.get("exchange", {}).get("testnet", True)

            self.log(
                "INFO",
                f"Exchange: {exchange_name} ({'TESTNET' if testnet else 'MAINNET'})",
            )

            if not testnet:
                self.log("WARNING", "Exchange is configured for MAINNET (live trading)")

            # 6. Validate trading symbols
            symbols = config.get("trading", {}).get("symbols", [])
            if not symbols:
                self.log("ERROR", "No trading symbols configured")
                return False

            self.log("INFO", f"Trading symbols: {', '.join(symbols)}")

            return len(self.errors) == 0

        except FileNotFoundError:
            self.log("ERROR", "crypto_trading/config.yaml not found")
            return False
        except Exception as e:
            self.log("ERROR", f"Failed to load crypto config: {e}")
            return False

    def validate_deployment_manifests(self) -> bool:
        """Validate deployment manifests"""
        print("\n" + "=" * 60)
        print("VALIDATING DEPLOYMENT MANIFESTS")
        print("=" * 60)

        success = True

        # IBKR manifest
        try:
            ibkr_manifest_path = (
                Path(__file__).parent
                / "execution"
                / "strategy_models"
                / "deployment_manifest.json"
            )
            with open(ibkr_manifest_path, "r") as f:
                ibkr_manifest = json.load(f)

            self.log(
                "INFO", f"IBKR manifest: {len(ibkr_manifest.get('items', []))} items"
            )

            # Validate each item
            for item in ibkr_manifest.get("items", []):
                path = Path(__file__).parent / item["path"]
                if not path.exists():
                    self.log("WARNING", f"IBKR: File not found: {item['path']}")

        except FileNotFoundError:
            self.log("WARNING", "IBKR deployment_manifest.json not found")
            success = False
        except Exception as e:
            self.log("ERROR", f"Failed to validate IBKR manifest: {e}")
            success = False

        # Crypto manifest
        try:
            crypto_manifest_path = (
                Path(__file__).parent
                / "crypto_trading"
                / "models"
                / "deployment_manifest.json"
            )
            with open(crypto_manifest_path, "r") as f:
                crypto_manifest = json.load(f)

            self.log(
                "INFO",
                f"Crypto manifest: {len(crypto_manifest.get('items', []))} items",
            )

            # Validate each item
            for item in crypto_manifest.get("items", []):
                path = Path(__file__).parent / item["path"]
                if not path.exists():
                    self.log("WARNING", f"Crypto: File not found: {item['path']}")

        except FileNotFoundError:
            self.log("WARNING", "Crypto deployment_manifest.json not found")
            success = False
        except Exception as e:
            self.log("ERROR", f"Failed to validate crypto manifest: {e}")
            success = False

        return success

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        print(f"✅ Passed: {len(self.info)} checks")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")

        if len(self.errors) == 0:
            print("\n🎉 Configuration is VALID and ready for production!")
            return True
        else:
            print("\n❌ Configuration has ERRORS and needs attention!")
            return False


def main():
    """Main validation routine"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate trading system configuration"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("TRADING SYSTEM CONFIGURATION VALIDATOR")
    print("=" * 60)

    validator = ConfigValidator(verbose=args.verbose)

    # Run validations
    ibkr_valid = validator.validate_ibkr_config()
    crypto_valid = validator.validate_crypto_config()
    manifest_valid = validator.validate_deployment_manifests()

    # Print summary
    all_valid = validator.print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
