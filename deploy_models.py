#!/usr/bin/env python3
"""
Deploy latest models, scalers, and configs from backtesting to live trading.
Ensures consistency between backtest and live trading environments.
Supports both stock (IBKR) and crypto assets.
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
import pickle
import sys
import yaml

# Add test_trained_models to path for utils
sys.path.insert(0, str(Path(__file__).parent / "test_trained_models"))
try:
    from utils import detect_asset_type, normalize_symbol, get_all_symbols
except ImportError:
    # Fallback if utils not available
    def detect_asset_type(symbol):
        crypto_indicators = ["-USD", "/USDT", ":USDT", "BTC", "ETH", "SOL", "DOGE"]
        return (
            "crypto"
            if any(ind in symbol.upper() for ind in crypto_indicators)
            else "stock"
        )

    def normalize_symbol(symbol):
        if ":" in symbol:
            symbol = symbol.split(":")[0]
        return symbol.replace("/", "-")

    def get_all_symbols():
        stock_symbols = []
        crypto_symbols = []
        try:
            sys.path.insert(0, "execution")
            from config import SYMBOLS

            stock_symbols = SYMBOLS
        except:
            pass
        try:
            config_path = Path("crypto_trading/config.yaml")
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                crypto_symbols = config.get("trading", {}).get("symbols", [])
        except:
            pass
        return stock_symbols, crypto_symbols


class PortfolioDeployer:
    """
    Reads a portfolio JSON file and orchestrates the full deployment pipeline:
    parse weights -> discover models -> generate config -> deploy files.
    """

    def __init__(self, portfolio_json_path: str):
        self.json_path = Path(portfolio_json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"Portfolio JSON not found: {self.json_path}")

        with open(self.json_path, "r") as f:
            self.weights = json.load(f)

        if not self.weights:
            raise ValueError(f"Portfolio JSON is empty: {self.json_path}")

        for ticker, weight in self.weights.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"Weight for {ticker} must be a number, got {type(weight).__name__}"
                )
            if weight < 0:
                raise ValueError(
                    f"Weight for {ticker} is negative ({weight}). All weights must be >= 0."
                )

        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.05:
            print(
                f"  WARNING: Portfolio weights sum to {total:.4f} (expected ~1.0). Proceeding anyway."
            )

        self.model_assignments = {}

    def discover_models(self, model_source_dir: Path = None):
        """
        Scan model_dumps for trained models matching each portfolio ticker.
        Assigns strategy_type='ml_signal' if model found, 'buy_and_hold' if not.
        """
        if model_source_dir is None:
            model_source_dir = Path("algos/model_dumps")

        print(f"\nScanning for trained models in {model_source_dir}...")

        for ticker in self.weights:
            search_patterns = [f"*_{ticker}_*.pkl", f"*_{ticker}_*.keras"]
            model_files = []
            for pattern in search_patterns:
                model_files.extend(list(model_source_dir.glob(pattern)))

            model_files = sorted(
                set(model_files), key=lambda x: x.stat().st_mtime, reverse=True
            )

            if model_files:
                latest_model = model_files[0]
                model_type = self._detect_model_type(latest_model.name)
                model_ext = latest_model.suffix

                self.model_assignments[ticker] = {
                    "strategy_type": "ml_signal",
                    "model_type": model_type,
                    "source_model_path": str(latest_model),
                    "model_extension": model_ext,
                }
                print(f"  {ticker}: ml_signal ({model_type}) <- {latest_model.name}")
            else:
                self.model_assignments[ticker] = {
                    "strategy_type": "buy_and_hold",
                    "model_type": None,
                    "source_model_path": None,
                    "model_extension": None,
                }
                print(f"  {ticker}: buy_and_hold (no trained model found)")

        ml_count = sum(
            1
            for a in self.model_assignments.values()
            if a["strategy_type"] == "ml_signal"
        )
        bah_count = len(self.model_assignments) - ml_count
        print(f"\nModel discovery: {ml_count} ml_signal, {bah_count} buy_and_hold")

    @staticmethod
    def _detect_model_type(model_filename: str) -> str:
        """Extract model type from filename using keyword matching."""
        name_lower = model_filename.lower()
        model_type_keywords = [
            ("lstm", "lstm"),
            ("svm", "svm"),
            ("li_reg", "li_reg"),
            ("linear", "li_reg"),
            ("dqn", "dqn"),
            ("xgb", "xgboost"),
            ("xgboost", "xgboost"),
            ("arima", "arima"),
            ("tcn", "tcn"),
            ("gbm", "gbm"),
            ("cnn", "cnn"),
            ("dnn", "dnn"),
            ("gnb", "gnb"),
            ("var", "var"),
            ("rf", "rf"),
            ("ensemble", "ensemble"),
            ("kmeans", "kmeans"),
        ]
        for keyword, model_type in model_type_keywords:
            if keyword in name_lower:
                return model_type
        return model_filename.split("_")[0]

    def rewrite_config(self, config_path: Path = None):
        """
        Programmatically rewrite TARGET_ALLOCATION, ASSET_SPECIFIC_CONFIGS,
        and SYMBOLS in execution/config.py.
        """
        if config_path is None:
            config_path = Path("execution/config.py")

        content = config_path.read_text()
        lines = content.split("\n")

        # Replace TARGET_ALLOCATION (uncommented dict with commented contents)
        new_target_alloc = self._generate_target_allocation()
        lines = self._replace_block(
            lines, "TARGET_ALLOCATION = {", new_target_alloc, allow_commented=False
        )

        # Replace ASSET_SPECIFIC_CONFIGS (entirely commented out block)
        new_asset_configs = self._generate_asset_specific_configs()
        lines = self._replace_block(
            lines, "ASSET_SPECIFIC_CONFIGS = {", new_asset_configs, allow_commented=True
        )

        # Uncomment SYMBOLS line
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "# SYMBOLS = list(ASSET_SPECIFIC_CONFIGS.keys())":
                lines[i] = "SYMBOLS = list(ASSET_SPECIFIC_CONFIGS.keys())"
                break
            if stripped == "SYMBOLS = list(ASSET_SPECIFIC_CONFIGS.keys())":
                break  # Already uncommented

        config_path.write_text("\n".join(lines))
        print(f"\n  Config rewritten: {config_path}")

    @staticmethod
    def _replace_block(lines, block_start_pattern, new_block, allow_commented=False):
        """
        Replace a Python dict block in the config file.
        Finds block start, tracks brace depth to find end, replaces entire block.
        """
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if block_start_pattern in stripped:
                start_idx = i
                break
            if allow_commented and f"# {block_start_pattern}" in stripped:
                start_idx = i
                break

        if start_idx is None:
            return lines

        depth = 0
        in_block = False
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if allow_commented:
                effective = line.lstrip()
                if effective.startswith("#"):
                    effective = effective.lstrip("#").lstrip()
            else:
                effective = line

            for ch in effective:
                if ch == "{":
                    depth += 1
                    in_block = True
                elif ch == "}":
                    depth -= 1
                    if in_block and depth == 0:
                        end_idx = i
                        break
            if end_idx is not None:
                break

        if end_idx is None:
            return lines

        new_lines = new_block.split("\n")
        return lines[:start_idx] + new_lines + lines[end_idx + 1 :]

    def _generate_target_allocation(self):
        """Generate TARGET_ALLOCATION dict as Python source code."""
        result_lines = ["TARGET_ALLOCATION = {"]
        sorted_tickers = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_tickers:
            result_lines.append(f'    "{ticker}": {weight},')
        result_lines.append("}")
        return "\n".join(result_lines)

    def _generate_asset_specific_configs(self):
        """Generate ASSET_SPECIFIC_CONFIGS dict as Python source code."""
        result_lines = ["ASSET_SPECIFIC_CONFIGS = {"]
        for ticker in sorted(self.model_assignments.keys()):
            assignment = self.model_assignments[ticker]
            strategy_type = assignment["strategy_type"]
            model_type = assignment["model_type"]

            if strategy_type == "ml_signal" and model_type:
                ext = assignment.get("model_extension", ".pkl")
                model_path = f"strategy_models/{ticker}_trading_model_{model_type}{ext}"
                result_lines.append(f'    "{ticker}": {{')
                result_lines.append('        "kelly_fraction": 2.0,')
                result_lines.append('        "strategy_type": "ml_signal",')
                result_lines.append(
                    f'        "model_type": "{model_type}",  # Auto-assigned by portfolio deployer'
                )
                result_lines.append(
                    f'        "strategy_model_path": "{model_path}",  # Auto-assigned by portfolio deployer'
                )
                result_lines.append('        "sequence_length": 5,')
                result_lines.append('        "lags": 5,')
                result_lines.append('        "min_position_shares": None,')
                result_lines.append("    },")
            else:
                result_lines.append(f'    "{ticker}": {{')
                result_lines.append('        "kelly_fraction": 2.0,')
                result_lines.append('        "strategy_type": "buy_and_hold",')
                result_lines.append('        "lags": 5,')
                result_lines.append("    },")

        result_lines.append("}")
        return "\n".join(result_lines)

    def dry_run(self, config_path: Path = None):
        """
        Show what would happen without modifying any files.

        Args:
            config_path: Path to config.py (for display purposes only).
        """
        if config_path is None:
            config_path = Path("execution/config.py")

        print("\n" + "=" * 60)
        print("DRY RUN - No files will be modified")
        print("=" * 60)

        print(f"\nSource: {self.json_path}")
        print(f"Target: {config_path}")
        print(f"Tickers: {len(self.weights)}")
        print(f"Weights sum: {sum(self.weights.values()):.4f}")

        print(f"\n{'Ticker':<20} {'Weight':>8} {'Strategy':<15} {'Model':<12}")
        print("-" * 60)

        for ticker in sorted(self.weights.keys()):
            weight = self.weights[ticker]
            assignment = self.model_assignments.get(ticker, {})
            strategy = assignment.get("strategy_type", "unknown")
            model = assignment.get("model_type", "-") or "-"
            print(f"{ticker:<20} {weight:>8.4f} {strategy:<15} {model:<12}")

        ml_count = sum(
            1
            for a in self.model_assignments.values()
            if a["strategy_type"] == "ml_signal"
        )
        bah_count = len(self.model_assignments) - ml_count

        print(f"\nSummary: {ml_count} ml_signal, {bah_count} buy_and_hold")

        print("\nGenerated TARGET_ALLOCATION:")
        print(self._generate_target_allocation())

        print("\nGenerated ASSET_SPECIFIC_CONFIGS (first 30 lines):")
        config_preview = self._generate_asset_specific_configs()
        preview_lines = config_preview.split("\n")[:30]
        print("\n".join(preview_lines))
        if len(config_preview.split("\n")) > 30:
            print(f"  ... ({len(config_preview.split(chr(10)))} total lines)")

        print("\n" + "=" * 60)
        print("DRY RUN complete. No files were modified.")
        print("Run without --dry-run to apply changes.")
        print("=" * 60)


def deploy_models_and_scalers():
    """
    Copy latest models, scalers, and configs to live trading directories.
    Automatically detects and handles both stock and crypto assets.
    """
    print("\n" + "=" * 60)
    print("MODEL AND SCALER DEPLOYMENT (STOCK & CRYPTO)")
    print("=" * 60)

    # Define source directories
    model_source_dir = Path("algos/model_dumps")
    scaler_source_dir = Path("algos/scalers")

    # Get all symbols from both configs
    stock_symbols, crypto_symbols = get_all_symbols()
    ALL_SYMBOLS = stock_symbols + crypto_symbols

    if not ALL_SYMBOLS:
        print("Warning: No symbols found in configs, using defaults")
        ALL_SYMBOLS = ["NVDA", "AVGO", "BTC-USD"]

    deployed_items = []

    print(f"\nStock symbols: {stock_symbols}")
    print(f"Crypto symbols: {crypto_symbols}")
    print(f"\nDeploying models and scalers for all symbols...")

    for symbol in ALL_SYMBOLS:
        asset_type = detect_asset_type(symbol)
        normalized_sym = normalize_symbol(symbol)

        # Determine deployment directory based on asset type
        if asset_type == "crypto":
            deployment_dir = Path("crypto_trading/models")
        else:
            deployment_dir = Path("execution/strategy_models")

        deployment_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Deploying {symbol} ({asset_type}) ---")
        print(f"    Normalized: {normalized_sym}")
        print(f"    Deploy to: {deployment_dir}")

        # Find and deploy latest model - handle different naming conventions
        # For crypto, also check normalized symbol and yfinance format
        search_patterns = [
            f"*_{symbol}_*.keras",
            f"*_{symbol}_*.pkl",
            f"*_{normalized_sym}_*.keras",
            f"*_{normalized_sym}_*.pkl",
        ]

        # For crypto assets, also search for yfinance format (e.g., BTC-USD)
        if asset_type == "crypto":
            # Convert BTC/USDT to BTC-USD for yfinance format
            base_symbol = (
                symbol.split("/")[0] if "/" in symbol else symbol.split("-")[0]
            )
            yfinance_symbol = f"{base_symbol}-USD"
            search_patterns.extend(
                [f"*_{yfinance_symbol}_*.keras", f"*_{yfinance_symbol}_*.pkl"]
            )

        model_files = []
        for pattern in search_patterns:
            model_files.extend(list(model_source_dir.glob(pattern)))

        # Remove duplicates
        model_files = list(set(model_files))

        if model_files:
            # Sort by modification time to get latest
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

            # Extract model type from filename
            model_name = latest_model.name
            if "lstm" in model_name.lower():
                model_type = "lstm"
            elif "svm" in model_name.lower():
                model_type = "svm"
            elif "li_reg" in model_name.lower() or "linear" in model_name.lower():
                model_type = "li_reg"
            elif "dqn" in model_name.lower():
                model_type = "dqn"
            elif "xgb" in model_name.lower() or "xgboost" in model_name.lower():
                model_type = "xgboost"
            elif "arima" in model_name.lower():
                model_type = "arima"
            elif "tcn" in model_name.lower():
                model_type = "tcn"
            elif "gbm" in model_name.lower():
                model_type = "gbm"
            else:
                # Try to extract from the beginning of filename
                model_type = model_name.split("_")[0]

            # Copy model to deployment directory with normalized name
            dest_path = (
                deployment_dir / f"{normalized_sym}_trading_model_{model_type}.pkl"
            )
            if latest_model.suffix == ".keras":
                # For Keras models, keep the .keras extension
                dest_path = (
                    deployment_dir
                    / f"{normalized_sym}_trading_model_{model_type}.keras"
                )

            shutil.copy2(latest_model, dest_path)
            print(f"  ✓ Deployed model: {latest_model.name} -> {dest_path.name}")
            deployed_items.append(("model", symbol, model_type, dest_path, asset_type))

        else:
            print(f"  ⚠ No model found for {symbol} or {normalized_sym}")
            model_type = "unknown"

        # Find and deploy latest scaler - check both original and normalized symbol
        scaler_search_patterns = [
            f"{model_type}_scaler_{symbol}_latest.pkl",
            f"{model_type}_scaler_{normalized_sym}_latest.pkl",
            f"scaler_{symbol}_latest.pkl",
            f"scaler_{normalized_sym}_latest.pkl",
        ]

        # For crypto assets, add yfinance format patterns (e.g., BTC-USD for BTC/USDT)
        if asset_type == "crypto":
            base_symbol = (
                symbol.split("/")[0] if "/" in symbol else symbol.split("-")[0]
            )
            yfinance_symbol = f"{base_symbol}-USD"
            scaler_search_patterns.extend(
                [
                    f"{model_type}_scaler_{yfinance_symbol}_latest.pkl",
                    f"scaler_{yfinance_symbol}_latest.pkl",
                    f"*_scaler_{yfinance_symbol}_*.pkl",  # Timestamped versions
                ]
            )

        # Add wildcard patterns last (less specific)
        scaler_search_patterns.extend(
            [f"*_{symbol}_*.pkl", f"*_{normalized_sym}_*.pkl"]
        )

        scaler_file = None
        for pattern in scaler_search_patterns:
            if "*" in pattern:
                files = list(scaler_source_dir.glob(pattern))
                if files:
                    scaler_file = max(files, key=lambda x: x.stat().st_mtime)
                    break
            else:
                file_path = scaler_source_dir / pattern
                if file_path.exists():
                    scaler_file = file_path
                    break

        if scaler_file and scaler_file.exists():
            # Deploy scaler with normalized naming for compatibility
            # For crypto: use both exchange format and yfinance format
            if asset_type == "crypto":
                base_symbol = (
                    symbol.split("/")[0] if "/" in symbol else symbol.split("-")[0]
                )
                yfinance_symbol = f"{base_symbol}-USD"

                destinations = [
                    deployment_dir
                    / f"{model_type}_scaler_{normalized_sym}.pkl",  # BTC-USDT
                    deployment_dir
                    / f"{model_type}_scaler_{yfinance_symbol}.pkl",  # BTC-USD
                    deployment_dir / f"scaler_{normalized_sym}.pkl",
                    deployment_dir / f"{normalized_sym}_scaler.pkl",
                ]
            else:
                # For stocks: use standard naming
                destinations = [
                    deployment_dir / f"{model_type}_scaler_{normalized_sym}.pkl",
                    deployment_dir / f"scaler_{normalized_sym}.pkl",
                    deployment_dir / f"{normalized_sym}_scaler.pkl",
                ]

            for dest_path in destinations:
                shutil.copy2(scaler_file, dest_path)

            print(
                f"  ✓ Deployed scaler: {scaler_file.name} -> {len(destinations)} locations"
            )
            deployed_items.append(
                ("scaler", symbol, model_type, destinations[0], asset_type)
            )

            # Also copy the metadata JSON if it exists
            json_file = scaler_file.with_suffix(".json")
            if json_file.exists():
                # Deploy metadata for all naming conventions
                for dest_path in destinations[
                    :2
                ]:  # Just the first two to avoid clutter
                    json_dest = dest_path.with_suffix(".json")
                    shutil.copy2(json_file, json_dest)
                print(f"  ✓ Deployed scaler metadata: {json_file.name}")
        else:
            print(f"  ⚠ No scaler found for {symbol} or {normalized_sym}")

        # Find and deploy seed info
        seed_files = list(model_source_dir.glob(f"seed_info_*_{symbol}_*.json")) + list(
            model_source_dir.glob(f"seed_info_*_{normalized_sym}_*.json")
        )

        # For crypto, also check yfinance format
        if asset_type == "crypto":
            base_symbol = (
                symbol.split("/")[0] if "/" in symbol else symbol.split("-")[0]
            )
            yfinance_symbol = f"{base_symbol}-USD"
            seed_files.extend(
                list(model_source_dir.glob(f"seed_info_*_{yfinance_symbol}_*.json"))
            )

        if seed_files:
            latest_seed = max(seed_files, key=lambda x: x.stat().st_mtime)
            dest_path = deployment_dir / f"seed_info_{normalized_sym}.json"
            shutil.copy2(latest_seed, dest_path)
            print(f"  ✓ Deployed seed info: {latest_seed.name}")
            deployed_items.append(
                ("seed_info", symbol, model_type, dest_path, asset_type)
            )

        # Find and deploy ARIMA settings if model is ARIMA
        if model_type == "arima":
            arima_settings_files = list(
                model_source_dir.glob(f"arima_settings_{symbol}_*.json")
            ) + list(model_source_dir.glob(f"arima_settings_{normalized_sym}_*.json"))

            # For crypto, also check yfinance format
            if asset_type == "crypto":
                base_symbol = (
                    symbol.split("/")[0] if "/" in symbol else symbol.split("-")[0]
                )
                yfinance_symbol = f"{base_symbol}-USD"
                arima_settings_files.extend(
                    list(
                        model_source_dir.glob(
                            f"arima_settings_{yfinance_symbol}_*.json"
                        )
                    )
                )

            if arima_settings_files:
                latest_settings = max(
                    arima_settings_files, key=lambda x: x.stat().st_mtime
                )
                dest_path = deployment_dir / f"arima_settings_{normalized_sym}.json"
                shutil.copy2(latest_settings, dest_path)
                print(f"  ✓ Deployed ARIMA settings: {latest_settings.name}")
                deployed_items.append(
                    ("arima_settings", symbol, model_type, dest_path, asset_type)
                )
            else:
                print(f"  ⚠ No ARIMA settings found for {symbol} or {normalized_sym}")

        # Find and deploy feature metadata
        feature_meta_source = Path("algos/scalers")
        feature_meta_patterns = [
            f"feature_meta_*_{symbol}.json",
            f"feature_meta_*_{normalized_sym}.json",
        ]
        if asset_type == "crypto":
            base_symbol = (
                symbol.split("/")[0] if "/" in symbol else symbol.split("-")[0]
            )
            yfinance_symbol = f"{base_symbol}-USD"
            feature_meta_patterns.append(f"feature_meta_*_{yfinance_symbol}.json")

        feature_meta_files = []
        for pattern in feature_meta_patterns:
            feature_meta_files.extend(list(feature_meta_source.glob(pattern)))
        # Also check model_dumps directory
        for pattern in feature_meta_patterns:
            feature_meta_files.extend(list(model_source_dir.glob(pattern)))

        if feature_meta_files:
            latest_meta = max(feature_meta_files, key=lambda x: x.stat().st_mtime)
            dest_path = (
                deployment_dir / f"feature_meta_{model_type}_{normalized_sym}.json"
            )
            shutil.copy2(latest_meta, dest_path)
            print(f"  ✓ Deployed feature metadata: {latest_meta.name}")
            deployed_items.append(
                ("feature_meta", symbol, model_type, dest_path, asset_type)
            )

    # Update configs with deployed models
    update_configs_with_deployments(deployed_items)

    # Create deployment manifest
    create_deployment_manifest(deployed_items)

    # Verify deployment
    verify_deployment(ALL_SYMBOLS)

    return deployed_items


def update_configs_with_deployments(deployed_items):
    """
    Updates appropriate config files with the deployed model paths.
    Handles both IBKR config.py and crypto config.yaml.
    """
    print("\n" + "-" * 40)
    print("Updating configs with deployment info...")

    # Group deployments by asset type
    stock_deployments = [
        item for item in deployed_items if item[4] == "stock" and item[0] == "model"
    ]
    crypto_deployments = [
        item for item in deployed_items if item[4] == "crypto" and item[0] == "model"
    ]

    # Update IBKR config for stock assets
    if stock_deployments:
        update_ibkr_config(stock_deployments)

    # Update crypto config for crypto assets
    if crypto_deployments:
        update_crypto_config(crypto_deployments)


def update_ibkr_config(deployments):
    """Update IBKR config.py with stock model deployments."""
    config_file = Path("execution/config.py")
    print(f"\nUpdating IBKR config: {config_file}")

    if not config_file.exists():
        print(f"  ⚠ Config file not found: {config_file}")
        return

    try:
        # Read the current config
        with open(config_file, "r") as f:
            lines = f.readlines()

        # Update each symbol's configuration
        for item in deployments:
            symbol = item[1]
            model_type = item[2]
            model_path = item[3]

            # Find the symbol's section in ASSET_SPECIFIC_CONFIGS
            in_symbol_block = False
            for i, line in enumerate(lines):
                if f"'{symbol}'" in line and "{" in line:
                    in_symbol_block = True
                elif in_symbol_block:
                    if "'model_type':" in line:
                        lines[i] = (
                            f"        'model_type': '{model_type}',  # Auto-updated by deploy_models.py\n"
                        )
                    elif "'strategy_model_path':" in line:
                        relative_path = model_path.relative_to(Path("execution"))
                        lines[i] = (
                            f"        'strategy_model_path': '{relative_path}',  # Updated by deploy_models.py\n"
                        )
                    elif "}," in line:
                        in_symbol_block = False

        # Write the updated config
        with open(config_file, "w") as f:
            f.writelines(lines)

        print(f"  ✓ Updated {len(deployments)} symbol(s) in IBKR config")

    except Exception as e:
        print(f"  ✗ Error updating IBKR config: {e}")


def update_crypto_config(deployments):
    """Update crypto config.yaml with crypto model deployments."""
    config_file = Path("crypto_trading/config.yaml")
    print(f"\nUpdating crypto config: {config_file}")

    if not config_file.exists():
        print(f"  ⚠ Config file not found: {config_file}")
        return

    try:
        # Load existing config
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Update ticker models
        if "strategy" not in config:
            config["strategy"] = {}
        if "ticker_models" not in config["strategy"]:
            config["strategy"]["ticker_models"] = {}

        for item in deployments:
            symbol = item[1]
            model_type = item[2]
            config["strategy"]["ticker_models"][symbol] = model_type
            print(f"  ✓ Updated {symbol} to use {model_type} model")

        # Write updated config
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print("  ✓ Crypto config updated successfully")

    except Exception as e:
        print(f"  ✗ Error updating crypto config: {e}")


def create_deployment_manifest(deployed_items):
    """Create deployment manifests in both deployment directories."""
    print("\n" + "-" * 40)
    print("Creating deployment manifests...")

    # Group by asset type
    stock_items = [item for item in deployed_items if item[4] == "stock"]
    crypto_items = [item for item in deployed_items if item[4] == "crypto"]

    # Create IBKR manifest
    if stock_items:
        manifest = {
            "deployed_at": datetime.now().isoformat(),
            "asset_type": "stock",
            "items": [
                {
                    "type": item[0],
                    "symbol": item[1],
                    "model_type": item[2],
                    "path": str(item[3]),
                }
                for item in stock_items
            ],
        }

        manifest_path = Path("execution/strategy_models/deployment_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"✓ IBKR deployment manifest saved to {manifest_path}")

    # Create crypto manifest
    if crypto_items:
        manifest = {
            "deployed_at": datetime.now().isoformat(),
            "asset_type": "crypto",
            "items": [
                {
                    "type": item[0],
                    "symbol": item[1],
                    "model_type": item[2],
                    "path": str(item[3]),
                }
                for item in crypto_items
            ],
        }

        manifest_path = Path("crypto_trading/models/deployment_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"✓ Crypto deployment manifest saved to {manifest_path}")

    # Create combined report
    report_path = Path("deployment_report.json")
    report = {
        "timestamp": datetime.now().isoformat(),
        "stock_deployments": len(stock_items),
        "crypto_deployments": len(crypto_items),
        "details": {
            "stock": [
                {
                    "symbol": item[1],
                    "type": item[0],
                    "model_type": item[2] if item[0] == "model" else None,
                }
                for item in stock_items
            ],
            "crypto": [
                {
                    "symbol": item[1],
                    "type": item[0],
                    "model_type": item[2] if item[0] == "model" else None,
                }
                for item in crypto_items
            ],
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Combined deployment report saved to {report_path}")
    print(f"  - Stock deployments: {len(stock_items)}")
    print(f"  - Crypto deployments: {len(crypto_items)}")


def verify_deployment(symbols):
    """Verify that deployments were successful."""
    print("\n" + "=" * 60)
    print("DEPLOYMENT VERIFICATION")
    print("=" * 60)

    for symbol in symbols:
        asset_type = detect_asset_type(symbol)
        normalized_sym = normalize_symbol(symbol)

        if asset_type == "crypto":
            deployment_dir = Path("crypto_trading/models")
        else:
            deployment_dir = Path("execution/strategy_models")

        print(f"\n{symbol} ({asset_type}):")

        # Check for model
        model_files = list(deployment_dir.glob(f"{normalized_sym}_trading_model_*"))
        model_exists = len(model_files) > 0

        # Determine model type from filename if model exists
        model_type = None
        if model_exists and model_files:
            model_name = model_files[0].name
            if "arima" in model_name.lower():
                model_type = "arima"

        # Check for scalers (search multiple naming conventions)
        scaler_patterns = [
            f"*scaler_{normalized_sym}.pkl",
            f"*scaler_{symbol}.pkl",
        ]

        # For crypto, also check yfinance format
        if asset_type == "crypto":
            base_symbol = (
                symbol.split("/")[0] if "/" in symbol else symbol.split("-")[0]
            )
            yfinance_symbol = f"{base_symbol}-USD"
            scaler_patterns.append(f"*scaler_{yfinance_symbol}.pkl")

        scaler_exists = False
        for pattern in scaler_patterns:
            if any(deployment_dir.glob(pattern)):
                scaler_exists = True
                break

        if model_exists and scaler_exists:
            print("  ✓ Model and scaler deployed successfully")
        elif model_exists and model_type == "arima":
            print("  ✓ ARIMA model deployed (scaler not required)")
        elif model_exists and model_type in ["unknown", None]:
            print("  ⚠ Model deployed (scaler requirement unknown)")
        elif model_exists:
            print("  ⚠ Model deployed but scaler missing")
        elif scaler_exists:
            print("  ⚠ Scaler deployed but model missing")
        else:
            print("  ✗ Neither model nor scaler found")

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print("=" * 60)


def deploy_carry_trade_models():
    """Deploy carry trade models for the cash portfolio engine.

    Finds the latest trained model for each carry trade pair configured in
    CASH_PORTFOLIO_CONFIG and copies it to execution/strategy_models/ with
    the carry_ prefix naming convention.
    """
    print("\n" + "=" * 60)
    print("CARRY TRADE MODEL DEPLOYMENT")
    print("=" * 60)

    try:
        sys.path.insert(0, "execution")
        from config import CASH_PORTFOLIO_CONFIG, CASH_REBALANCING_MODE
    except ImportError:
        print(
            "Warning: CASH_PORTFOLIO_CONFIG not found. Skipping carry trade deployment."
        )
        return

    if CASH_REBALANCING_MODE == "legacy":
        print("Cash rebalancing mode is 'legacy'. Skipping carry trade deployment.")
        return

    model_source_dir = Path("algos/model_dumps")
    deployment_dir = Path("execution/strategy_models")
    deployment_dir.mkdir(parents=True, exist_ok=True)

    # Support both old multi-pair config (carry_trade_models) and new single-pair (carry_model)
    carry_models = CASH_PORTFOLIO_CONFIG.get("carry_trade_models", {})
    if not carry_models:
        # New single-pair config: carry_model + carry_pair
        single_model = CASH_PORTFOLIO_CONFIG.get("carry_model")
        carry_pair = CASH_PORTFOLIO_CONFIG.get("carry_pair")
        if single_model and carry_pair:
            carry_models = {carry_pair: single_model}
        else:
            print("No carry trade models configured.")
            return

    for pair_ticker, model_cfg in carry_models.items():
        model_type = model_cfg.get("model_type", "gnb")
        target_path = deployment_dir / Path(model_cfg["strategy_model_path"]).name

        print(f"\n--- Deploying carry model for {pair_ticker} ({model_type}) ---")
        print(f"    Target: {target_path}")

        # Search for trained models matching this pair
        search_patterns = [
            f"{model_type}_*_{pair_ticker}_*.pkl",
            f"{model_type}*_{pair_ticker}_*.pkl",
            f"*{model_type}*{pair_ticker}*.pkl",
        ]

        model_files = []
        for pattern in search_patterns:
            model_files.extend(list(model_source_dir.glob(pattern)))

        model_files = list(set(model_files))

        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            print(f"    Source: {latest_model.name}")

            shutil.copy2(latest_model, target_path)
            print(f"    ✓ Deployed: {target_path}")

            # Also deploy scaler if exists
            scaler_source_dir = Path("algos/scalers")
            scaler_patterns = [
                f"{model_type}_scaler_{pair_ticker}_*.pkl",
                f"{model_type}*scaler*{pair_ticker}*.pkl",
            ]
            scaler_files = []
            for pattern in scaler_patterns:
                scaler_files.extend(list(scaler_source_dir.glob(pattern)))

            if scaler_files:
                latest_scaler = max(scaler_files, key=lambda x: x.stat().st_mtime)
                scaler_target = deployment_dir / f"carry_{pair_ticker}_scaler.pkl"
                shutil.copy2(latest_scaler, scaler_target)
                print(f"    ✓ Scaler: {scaler_target}")
        else:
            print(f"    ✗ No trained model found for {pair_ticker} ({model_type})")
            print(
                f"      Run: python algos/backtest_code/run_backtest_optimized.py "
                f"--model_name {model_type} --ticker {pair_ticker}"
            )


def main():
    """Main deployment function with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy models, scalers, and portfolio configs to live trading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy from a portfolio JSON (recommended workflow):
  python deploy_models.py --portfolio algos/backtest_code/data/portfolio_weights_max_sharpe_Q2_2026.json

  # Preview changes without modifying anything:
  python deploy_models.py --portfolio weights.json --dry-run

  # Legacy mode (no portfolio JSON, deploy models for existing config):
  python deploy_models.py
        """,
    )

    parser.add_argument(
        "--portfolio",
        type=str,
        default=None,
        help="Path to portfolio JSON file with {ticker: weight} mapping. "
        "Updates config.py and deploys models in one step.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying any files.",
    )

    parser.add_argument(
        "--bypass-trials-check",
        action="store_true",
        help=(
            "Skip the trials-ledger gate (Phase 3.4 of the Revision Protocol). "
            "ONLY for operational hot-fixes, NOT for strategy revisions. "
            "Audited: the bypass is logged with a timestamp + git hash."
        ),
    )

    args = parser.parse_args()

    # Revision Protocol — Phase 3.4 trials gate.
    # If the deployer is being invoked with a portfolio JSON, we require
    # an accepted trial in the ledger whose source_file matches that JSON
    # (or whose layer == "weights"). Bypass with --bypass-trials-check.
    if args.portfolio and not args.dry_run:
        try:
            from algos.wfov.trials_ledger import iter_trials, get_cumulative_n
            portfolio_basename = Path(args.portfolio).name
            accepted = list(iter_trials(layer="weights", accepted=True))
            relevant = [t for t in accepted
                        if portfolio_basename in (t.get("source_file") or "")
                        or portfolio_basename in (t.get("description") or "")]
            if not relevant and not args.bypass_trials_check:
                print(
                    "\n" + "=" * 60 + "\n"
                    "REVISION PROTOCOL: DEPLOY BLOCKED\n"
                    "=" * 60 + "\n"
                    f"No accepted 'weights' trial in algos/wfov/trials_ledger.db\n"
                    f"references portfolio: {portfolio_basename}\n\n"
                    f"Cumulative trials in ledger: {get_cumulative_n()}\n\n"
                    "To proceed, either:\n"
                    "  1. Run `python scripts/revision_check.py PROPOSAL.yaml --commit`\n"
                    "     against this portfolio change, or\n"
                    "  2. Re-invoke with --bypass-trials-check (audited)\n"
                    "=" * 60,
                    file=sys.stderr,
                )
                sys.exit(3)
            if args.bypass_trials_check:
                import subprocess
                ghash = ""
                try:
                    ghash = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        stderr=subprocess.DEVNULL, timeout=5
                    ).decode().strip()
                except Exception:
                    pass
                print(f"[WARN] Trials-check bypassed at {datetime.now().isoformat()} "
                      f"(git {ghash[:8]}). This deployment is operational, not a "
                      f"strategy revision.", file=sys.stderr)
        except ImportError:
            # Ledger not available; allow deployment but warn.
            print("[WARN] Trials ledger module not available; gate skipped.",
                  file=sys.stderr)

    if args.portfolio:
        # New portfolio-driven deployment flow
        deployer = PortfolioDeployer(args.portfolio)
        deployer.discover_models()

        if args.dry_run:
            deployer.dry_run()
            return

        # Rewrite config.py with new weights and asset configs
        deployer.rewrite_config()

        # NOTE: deploy_models_and_scalers() calls get_all_symbols() which uses
        # importlib.util.spec_from_file_location() to read config.py fresh from
        # disk each time (no sys.modules caching). So no cache flush is needed
        # after rewrite_config(). Defensive: clear stale "config" from
        # sys.modules in case the fallback code path is used.
        if "config" in sys.modules:
            del sys.modules["config"]

        # Now run the existing model/scaler deployment
        deploy_models_and_scalers()
        deploy_carry_trade_models()

        print("\n" + "=" * 60)
        print("PORTFOLIO DEPLOYMENT COMPLETE")
        print("=" * 60)
        print(f"Source: {args.portfolio}")
        print(f"Tickers: {len(deployer.weights)}")

        ml_count = sum(
            1
            for a in deployer.model_assignments.values()
            if a["strategy_type"] == "ml_signal"
        )
        bah_count = len(deployer.model_assignments) - ml_count
        print(f"ML Signal: {ml_count} | Buy & Hold: {bah_count}")
        print(f"\nNext: python execution/main.py --region ALL")
        print("=" * 60)
    else:
        if args.dry_run:
            print("Error: --dry-run requires --portfolio")
            sys.exit(1)

        # Legacy deployment (existing behavior)
        deploy_models_and_scalers()
        deploy_carry_trade_models()


if __name__ == "__main__":
    main()
