#!/usr/bin/env python3
"""
Automated Model Selection Workflow (Answer A: Maximum Returns)

End-to-end workflow from training to deployment recommendation.
Optimized for agile traders who monitor daily and can react quickly.

Philosophy:
- Screen out garbage (p > 0.20, negative Sharpe)
- Among valid models, pick HIGHEST SHARPE
- Flag risks (regime dependency) but don't auto-reject
- Trust trader to monitor and switch if needed

Author: jcp
Date: 2025-12-03
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
import yaml

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from algos.wfov.model_ranker import ModelRanker


def cleanup_old_artifacts(
    max_age_days: int = 30,
    dry_run: bool = False,
    verbose: bool = True,
):
    """
    Remove old WFOV results, model dumps, and intermediate data files.

    Args:
        max_age_days: Remove files older than this many days (default: 30)
        dry_run: If True, only report what would be deleted without deleting
        verbose: Print detailed info about files found
    """
    project_root = Path(__file__).parent.parent
    cutoff = datetime.now() - timedelta(days=max_age_days)
    cutoff_ts = cutoff.timestamp()

    cleanup_targets = [
        (
            "WFOV Summaries",
            project_root / "algos" / "wfov" / "results" / "summaries",
            "*.json",
        ),
        (
            "WFOV Iterations",
            project_root / "algos" / "wfov" / "results" / "iterations",
            "*.csv",
        ),
        ("WFOV Logs", project_root / "algos" / "wfov" / "results" / "logs", "*"),
        ("Intermediate Data", project_root / "algos" / "data", "*.csv"),
        ("Workflow Logs", project_root / "workflow_results", "*.txt"),
    ]

    total_files = 0
    total_bytes = 0

    print(f"\n{'=' * 80}")
    print(f"ARTIFACT CLEANUP {'(DRY RUN)' if dry_run else ''}")
    print(
        f"Removing files older than {max_age_days} days (before {cutoff.strftime('%Y-%m-%d')})"
    )
    print(f"{'=' * 80}")

    for label, directory, pattern in cleanup_targets:
        if not directory.exists():
            continue

        old_files = []
        for f in directory.glob(pattern):
            if f.is_file() and f.stat().st_mtime < cutoff_ts:
                old_files.append(f)

        if not old_files:
            print(f"\n  {label}: No old files found")
            continue

        dir_bytes = sum(f.stat().st_size for f in old_files)
        total_files += len(old_files)
        total_bytes += dir_bytes

        print(f"\n  {label}: {len(old_files)} files ({dir_bytes / 1024 / 1024:.1f} MB)")

        if not dry_run:
            for f in old_files:
                f.unlink()
            print(f"    Deleted.")
        else:
            if verbose and len(old_files) <= 5:
                for f in old_files:
                    print(f"    Would delete: {f.name}")
            elif verbose:
                print(f"    Would delete: {old_files[0].name} ... {old_files[-1].name}")

    # Model dumps cleanup (special: keep latest per model+ticker, remove old)
    model_dumps_dir = project_root / "algos" / "model_dumps"
    if model_dumps_dir.exists():
        old_model_files = [
            f
            for f in model_dumps_dir.iterdir()
            if f.is_file() and f.stat().st_mtime < cutoff_ts
        ]
        if old_model_files:
            model_bytes = sum(f.stat().st_size for f in old_model_files)
            total_files += len(old_model_files)
            total_bytes += model_bytes
            print(
                f"\n  Model Dumps: {len(old_model_files)} files ({model_bytes / 1024 / 1024:.1f} MB)"
            )
            if not dry_run:
                for f in old_model_files:
                    f.unlink()
                print(f"    Deleted.")
        else:
            print(f"\n  Model Dumps: No old files found")

    print(f"\n{'─' * 80}")
    action = "Would remove" if dry_run else "Removed"
    print(f"  {action}: {total_files} files ({total_bytes / 1024 / 1024:.1f} MB)")
    print(f"{'=' * 80}")


class DualLogger:
    """
    Dual output logger - writes to both console and file simultaneously.
    Avoids issues with tee command and subprocess buffering.
    """

    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.log_file = open(filepath, "w", buffering=1)  # Line buffering
        self.filepath = filepath

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def load_tickers_from_json(json_path: str) -> list:
    """
    Load ticker symbols from a JSON weights file.

    Expects a flat JSON object mapping ticker symbols to numeric weights
    (e.g., portfolio allocation files like mc_1_weights.json, hrp_weights.json).
    Only the keys (ticker symbols) are extracted; weights are ignored.

    Args:
        json_path: Path to the JSON file.

    Returns:
        List of ticker symbol strings (preserving insertion order).

    Raises:
        SystemExit: On file-not-found, invalid JSON, or unexpected format.
    """
    json_file = Path(json_path)

    if not json_file.exists():
        print(f"❌ Tickers JSON file not found: {json_file.resolve()}")
        sys.exit(1)

    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {json_file}: {e}")
        sys.exit(1)

    if not isinstance(data, dict):
        print(
            f"❌ Expected JSON object with ticker keys, got {type(data).__name__} "
            f"in {json_file}"
        )
        sys.exit(1)

    tickers = list(data.keys())

    if not tickers:
        print(f"❌ JSON file contains no tickers: {json_file}")
        sys.exit(1)

    return tickers


def load_model_config(config_path: str = "model_selection_config.yaml") -> dict:
    """
    Load model selection configuration from YAML file.

    Args:
        config_path: Path to config file (default: model_selection_config.yaml)

    Returns:
        Dict with default_models and ticker_models
    """
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"⚠️  Config file not found: {config_file}")
        print("   Using hardcoded defaults")
        return {
            "default_models": ["lstm", "svm_optimized", "xgb_optimized", "li_reg"],
            "ticker_models": {},
        }

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Validate config structure
        if "default_models" not in config:
            config["default_models"] = [
                "lstm",
                "svm_optimized",
                "xgb_optimized",
                "li_reg",
            ]

        if "ticker_models" not in config:
            config["ticker_models"] = {}

        return config

    except Exception as e:
        print(f"⚠️  Error loading config: {e}")
        print("   Using hardcoded defaults")
        return {
            "default_models": ["lstm", "svm_optimized", "xgb_optimized", "li_reg"],
            "ticker_models": {},
        }


def get_models_for_ticker(ticker: str, config: dict, preset: str = None) -> list:
    """
    Get model list for a specific ticker from config.

    Args:
        ticker: Ticker symbol
        config: Config dict from load_model_config()
        preset: Optional preset name ('quick', 'comprehensive')

    Returns:
        List of model names to test for this ticker
    """
    # Priority 1: Preset (if specified) - OVERRIDES EVERYTHING
    if preset:
        preset_key = f"{preset}_models"
        if preset_key in config and config[preset_key]:
            models = config[preset_key]
            # Validate it's a list and not empty
            if isinstance(models, list) and len(models) > 0:
                print(
                    f"✓ Using '{preset}' preset ({len(models)} models): {', '.join(models)}"
                )
                return models
            else:
                print(
                    f"⚠️  Preset '{preset}' is empty in config, falling back to normal selection"
                )
        else:
            print(
                f"⚠️  Preset '{preset}' not found in config, falling back to normal selection"
            )

    # Priority 2: Ticker-specific models
    ticker_models = config.get("ticker_models", {})
    if ticker in ticker_models:
        models = ticker_models[ticker]
        # Handle None or empty list (YAML parses empty/commented sections as None)
        if models and isinstance(models, list) and len(models) > 0:
            print(
                f"✓ Using ticker-specific models for {ticker} ({len(models)} models): {', '.join(models)}"
            )
            return models
        else:
            # Ticker defined but no models (all commented out) → fall back to default
            print(f"ℹ️  Ticker {ticker} in config but has no models, using default")

    # Priority 3: Default models
    models = config.get("default_models")
    if models and isinstance(models, list) and len(models) > 0:
        print(
            f"ℹ️  Using default models for {ticker} ({len(models)} models): {', '.join(models)}"
        )
        return models
    else:
        # Config missing or empty → hardcoded fallback
        models = ["lstm", "svm_optimized", "xgb_optimized", "li_reg"]
        print(
            f"⚠️  Config default_models empty, using hardcoded for {ticker}: {', '.join(models)}"
        )
        return models


class ModelSelectionWorkflow:
    """
    Automated workflow for model training, validation, and selection.
    """

    def __init__(
        self,
        ticker: str,
        candidate_models: list = None,
        wfov_iterations: int = 50,
        parallel: bool = True,
        use_config: bool = True,
        config_path: str = "model_selection_config.yaml",
        preset: str = None,
        quiet: bool = False,
        skip_training: bool = False,
        skip_validation: bool = False,
        interval: str = "1d",
        no_plots: bool = False,
        from_store: bool = False,
    ):
        """
        Initialize workflow for a ticker.

        Args:
            ticker: Ticker symbol
            candidate_models: List of model types to test (overrides config if provided)
            wfov_iterations: Number of WFOV iterations per model (default: 50 for speed)
            parallel: Run WFOV in parallel (default: True)
            use_config: Load models from config file (default: True)
            config_path: Path to config file (default: model_selection_config.yaml)
            preset: Preset model list ('quick', 'comprehensive')
            quiet: Suppress intermediate output (for batch mode)
            skip_training: Skip training step (use existing models)
            skip_validation: Skip WFOV validation (use existing results)
            interval: Data interval - '1d' (daily), '1wk' (weekly), '1mo' (monthly). Default: '1d'
            no_plots: Skip generating plots (still show output). Default: False
            from_store: Use parquet market data store only (no yfinance fallback).
        """
        self.ticker = ticker
        self.wfov_iterations = wfov_iterations
        self.parallel = parallel
        self.quiet = quiet
        self.skip_training = skip_training
        self.skip_validation = skip_validation
        self.interval = interval
        self.no_plots = no_plots
        self.from_store = from_store
        self.workflow_start_time = datetime.now()  # Track when workflow started

        # Load models from config or use provided list
        if candidate_models is not None:
            # Command line override takes precedence
            self.candidate_models = candidate_models
            model_source = "command line"
        elif use_config:
            # Load from config file
            config = load_model_config(config_path)
            self.candidate_models = get_models_for_ticker(ticker, config, preset=preset)
            model_source = f"config file{f' (preset: {preset})' if preset else ''}"
        else:
            # Hardcoded default
            self.candidate_models = ["lstm", "svm_optimized", "xgb_optimized", "li_reg"]
            model_source = "hardcoded default"

        # Load feature engineering info for logging
        self._feature_configs = {}  # model_name -> FeatureConfig
        self._feature_hash = None
        try:
            from algos.common.feature_engine import FeatureConfig

            # Load default config for this ticker (no model override) to get baseline
            fc_default = FeatureConfig(ticker=ticker)
            if fc_default.indicators:
                self._feature_hash = fc_default.config_hash
                # Load per-model configs to detect differences
                for model in self.candidate_models:
                    fc = FeatureConfig(model_name=model, ticker=ticker)
                    self._feature_configs[model] = fc
        except (ImportError, Exception):
            pass  # Feature engine not available

        if not quiet:
            print(f"\n{'=' * 100}")
            print(f"MODEL SELECTION WORKFLOW: {ticker}")
            print("=" * 100)
            print(f"Model Source: {model_source}")
            print(f"Candidates: {', '.join(self.candidate_models)}")
            print(f"WFOV Iterations: {wfov_iterations}")
            print(f"Parallel Execution: {parallel}")
            print(f"Data Interval: {interval}")
            print(f"From Store Only: {from_store}")

            # Feature engineering summary
            if self._feature_configs:
                # Show default config
                default_fc = next(iter(self._feature_configs.values()))
                print(f"Feature Engineering: {default_fc.describe()}")
                # Only mention models that differ from default
                default_n = len(default_fc.indicators)
                diffs = [
                    f"{m}({'+' if len(fc.indicators) > default_n else ''}{len(fc.indicators) - default_n})"
                    for m, fc in self._feature_configs.items()
                    if len(fc.indicators) != default_n
                    or fc.config_hash != default_fc.config_hash
                ]
                if diffs:
                    print(f"  Model overrides: {', '.join(diffs)}")

    def _subprocess_env(self) -> dict:
        """Build subprocess env with optional store-only mode."""
        env = os.environ.copy()
        if self.from_store:
            env["MARKET_DATA_STORE_ONLY"] = "1"
        # Prevent HDF5 file lock contention when multiple processes
        # save Keras models concurrently to the same directory.
        # Without this, concurrent .keras saves cause BlockingIOError or
        # OSError("file signature not found") from h5py.
        env["HDF5_USE_FILE_LOCKING"] = "FALSE"
        env["STRIP_PHANTOM_ROWS"] = "1"
        return env

    def _prewarm_data_cache(self, start_date: str, end_date: str):
        """
        Pre-download market data into disk cache so parallel subprocesses
        don't all hit the network independently.
        """
        try:
            from algos.common.data_cache import OptimizedDataLoader

            loader = OptimizedDataLoader()
            loader.load_data(self.ticker, start_date, end_date, self.interval)
        except Exception:
            pass  # Best-effort; subprocesses will download if cache miss

    def step1_train_candidates(self, lookback_days: int = 1260, max_parallel: int = 3):
        """
        Step 1: Train all candidate models on same data (in parallel).

        Args:
            lookback_days: Lookback period for training (default: 1260 = 5 years)
            max_parallel: Max concurrent training subprocesses (default: 3).
                          Set to 1 for serial execution. Keep low to avoid
                          CPU oversubscription (sklearn models use n_jobs=-1).
        """
        if self.skip_training:
            if not self.quiet:
                print(f"\n⏭️  Skipping training (using existing models)")
            return

        if not self.quiet:
            print(f"\n{'-' * 100}")
            print("STEP 1: Training Candidate Models")
            print("-" * 100)

        # Pre-warm data cache for parallel subprocesses
        from datetime import timedelta as _td

        _end = datetime.now().strftime("%Y-%m-%d")
        _start = (datetime.now() - _td(days=lookback_days)).strftime("%Y-%m-%d")
        self._prewarm_data_cache(_start, _end)

        # Build commands for all models
        model_cmds = []
        for model in self.candidate_models:
            cmd = [
                "python",
                "algos/backtest_code/run_backtest_optimized.py",
                "--model_name",
                model,
                "--ticker",
                self.ticker,
                "--lookback_days",
                str(lookback_days),
                "--interval",
                self.interval,
            ]

            # Suppress all file output -- workflow is decision-only
            cmd.append("--skip-model-save")
            cmd.append("--no-save-intermediates")
            cmd.append("--no-plots")

            # Add model-specific params if needed
            if model == "arima":
                cmd.extend(
                    [
                        "--arima_threshold",
                        "0.0002",
                        "--arima_zscore",
                        "1.0",
                        "--arima_lookback",
                        "5",
                    ]
                )

            model_cmds.append((model, cmd))

        trained_count = 0
        failed_count = 0

        failed_names = []

        if max_parallel > 1 and len(model_cmds) > 1:
            # Parallel training with bounded concurrency
            active_procs = []  # List of (model_name, Popen)
            pending = list(model_cmds)

            while pending or active_procs:
                # Launch new processes up to max_parallel
                while pending and len(active_procs) < max_parallel:
                    model, cmd = pending.pop(0)
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=self._subprocess_env(),
                    )
                    active_procs.append((model, proc))

                # Poll active processes for completion
                import time as _time

                still_active = []
                for model, proc in active_procs:
                    retcode = proc.poll()
                    if retcode is not None:
                        # Process finished — read stderr for diagnostics
                        _, stderr_bytes = proc.communicate()
                        if retcode == 0:
                            trained_count += 1
                        else:
                            failed_count += 1
                            failed_names.append(model)
                            if stderr_bytes and not self.quiet:
                                stderr_text = stderr_bytes.decode(
                                    "utf-8", errors="replace"
                                ).strip()
                                # Show last few lines of stderr for diagnosis
                                err_lines = stderr_text.splitlines()[-5:]
                                print(
                                    f"    {model} FAILED (exit {retcode}): {' | '.join(err_lines)}"
                                )
                    else:
                        still_active.append((model, proc))

                active_procs = still_active

                if active_procs:
                    _time.sleep(0.5)  # Brief sleep to avoid busy-waiting
        else:
            # Serial training (original behavior)
            for model, cmd in model_cmds:
                try:
                    if self.quiet:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=300,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            env=self._subprocess_env(),
                        )
                    else:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=300,
                            env=self._subprocess_env(),
                        )

                    if result.returncode == 0:
                        trained_count += 1
                    else:
                        failed_count += 1
                        failed_names.append(model)
                        # Surface the failure reason from subprocess stderr/stdout
                        if not self.quiet:
                            err_text = (result.stderr or "").strip()
                            if not err_text:
                                err_text = (result.stdout or "").strip()
                            if err_text:
                                err_lines = err_text.splitlines()[-5:]
                                print(
                                    f"    {model} FAILED (exit {result.returncode}): {' | '.join(err_lines)}"
                                )
                            else:
                                print(
                                    f"    {model} FAILED (exit {result.returncode}): no error output captured"
                                )
                except subprocess.TimeoutExpired:
                    failed_count += 1
                    failed_names.append(model)
                    if not self.quiet:
                        print(f"    {model} FAILED: training timeout (>300s)")
                except Exception as e:
                    failed_count += 1
                    failed_names.append(model)
                    if not self.quiet:
                        print(f"    {model} FAILED: {e}")

        if not self.quiet:
            total = trained_count + failed_count
            if failed_count == 0:
                print(f"  Trained {trained_count}/{total} models successfully")
            else:
                print(
                    f"  Trained {trained_count}/{total} models ({failed_count} failed: {', '.join(failed_names)})"
                )

    def step2_validate_with_wfov(
        self,
        mode: str = "monte_carlo",
        seed: int = 42,
        start_date: str = "2020-01-01",
        end_date: str = None,
        min_lookback_days: int = 365,
        max_lookback_days: int = 1825,
        initial_train_days: int = None,
        window_size: int = None,
        test_days: int = None,
        step_days: int = None,
    ):
        """
        Step 2: Run WFOV validation on all candidates.

        Args:
            mode: Validation mode ('monte_carlo', 'walk_forward_expanding', 'walk_forward_rolling')
            seed: Random seed for reproducibility
            start_date: Start date for WFOV
            end_date: End date for WFOV (default: today)
            min_lookback_days: Min lookback for Monte Carlo
            max_lookback_days: Max lookback for Monte Carlo
            initial_train_days: Initial training days for walk-forward expanding
            window_size: Window size for walk-forward rolling
            test_days: Test period size for walk-forward modes
            step_days: Step size for walk-forward modes
        """
        if self.skip_validation:
            if not self.quiet:
                print(f"\n⏭️  Skipping validation (using existing WFOV results)")
            return

        from datetime import datetime as dt

        if end_date is None:
            end_date = dt.now().strftime("%Y-%m-%d")

        if not self.quiet:
            print(f"\n{'-' * 100}")
            print(f"STEP 2: WFOV Validation ({mode.upper()})")
            print("-" * 100)

        # Pre-warm data cache for parallel WFOV subprocesses
        self._prewarm_data_cache(start_date, end_date)

        if self.parallel and mode == "monte_carlo":
            if not self.quiet:
                print("Running validations in PARALLEL (faster)...")

            # Build commands for all models first
            model_cmds = []
            for model in self.candidate_models:
                cmd = [
                    "python",
                    "-m",
                    "algos.wfov.wfov_runner",
                    "--mode",
                    mode,
                    "--model_name",
                    model,
                    "--ticker",
                    self.ticker,
                    "--start_date",
                    start_date,
                    "--end_date",
                    end_date,
                    "--seed",
                    str(seed),
                    "--max_workers",
                    "1",
                    "--interval",
                    self.interval,
                ]

                # Mode-specific parameters
                if mode == "monte_carlo":
                    cmd.extend(
                        [
                            "--iterations",
                            str(self.wfov_iterations),
                            "--min_lookback_days",
                            str(min_lookback_days),
                            "--max_lookback_days",
                            str(max_lookback_days),
                        ]
                    )
                elif mode == "walk_forward_expanding":
                    if (
                        initial_train_days is None
                        or test_days is None
                        or step_days is None
                    ):
                        raise ValueError(
                            "Walk-forward expanding requires: initial_train_days, test_days, step_days"
                        )
                    cmd.extend(
                        [
                            "--initial_train_days",
                            str(initial_train_days),
                            "--test_days",
                            str(test_days),
                            "--step_days",
                            str(step_days),
                        ]
                    )
                elif mode == "walk_forward_rolling":
                    if window_size is None or test_days is None or step_days is None:
                        raise ValueError(
                            "Walk-forward rolling requires: window_size, test_days, step_days"
                        )
                    cmd.extend(
                        [
                            "--window_size",
                            str(window_size),
                            "--test_days",
                            str(test_days),
                            "--step_days",
                            str(step_days),
                        ]
                    )

                # Add model-specific params
                if model == "arima":
                    cmd.extend(
                        [
                            "--arima_threshold",
                            "0.0002",
                            "--arima_zscore",
                            "1.0",
                            "--arima_lookback",
                            "5",
                        ]
                    )

                # Suppress all file output except summary JSON -- workflow is decision-only
                cmd.append("--no-save-iterations")
                cmd.append("--no-plots")

                model_cmds.append((model, cmd))

            # Launch with bounded concurrency (max 3 concurrent)
            max_parallel = 3
            validated_count = 0
            val_failed_count = 0
            val_failed_names = []
            active_procs = []
            pending = list(model_cmds)

            while pending or active_procs:
                # Launch new processes up to max_parallel
                while pending and len(active_procs) < max_parallel:
                    model, cmd = pending.pop(0)
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=self._subprocess_env(),
                    )
                    active_procs.append((model, proc))

                # Poll active processes for completion
                import time as _time

                still_active = []
                for model, proc in active_procs:
                    retcode = proc.poll()
                    if retcode is not None:
                        if retcode == 0:
                            validated_count += 1
                        else:
                            val_failed_count += 1
                            val_failed_names.append(model)
                    else:
                        still_active.append((model, proc))

                active_procs = still_active

                if active_procs:
                    _time.sleep(0.5)

        else:
            validated_count = 0
            val_failed_count = 0
            val_failed_names = []

            for model in self.candidate_models:
                cmd = [
                    "python",
                    "-m",
                    "algos.wfov.wfov_runner",
                    "--mode",
                    mode,
                    "--model_name",
                    model,
                    "--ticker",
                    self.ticker,
                    "--start_date",
                    start_date,
                    "--end_date",
                    end_date,
                    "--seed",
                    str(seed),
                    "--interval",
                    self.interval,
                ]

                # Mode-specific parameters
                if mode == "monte_carlo":
                    cmd.extend(
                        [
                            "--iterations",
                            str(self.wfov_iterations),
                            "--min_lookback_days",
                            str(min_lookback_days),
                            "--max_lookback_days",
                            str(max_lookback_days),
                        ]
                    )
                elif mode == "walk_forward_expanding":
                    cmd.extend(
                        [
                            "--initial_train_days",
                            str(initial_train_days),
                            "--test_days",
                            str(test_days),
                            "--step_days",
                            str(step_days),
                        ]
                    )
                elif mode == "walk_forward_rolling":
                    cmd.extend(
                        [
                            "--window_size",
                            str(window_size),
                            "--test_days",
                            str(test_days),
                            "--step_days",
                            str(step_days),
                        ]
                    )

                # Add model-specific params
                if model == "arima":
                    cmd.extend(
                        [
                            "--arima_threshold",
                            "0.0002",
                            "--arima_zscore",
                            "1.0",
                            "--arima_lookback",
                            "5",
                        ]
                    )

                # Suppress all file output except summary JSON -- workflow is decision-only
                cmd.append("--no-save-iterations")
                cmd.append("--no-plots")

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        env=self._subprocess_env(),
                    )
                    if result.returncode == 0:
                        validated_count += 1
                    else:
                        val_failed_count += 1
                        val_failed_names.append(model)
                except subprocess.TimeoutExpired:
                    val_failed_count += 1
                    val_failed_names.append(model)

        if not self.quiet:
            total = validated_count + val_failed_count
            if val_failed_count == 0:
                print(f"  Validated {validated_count}/{total} models successfully")
            else:
                print(
                    f"  Validated {validated_count}/{total} models ({val_failed_count} failed: {', '.join(val_failed_names)})"
                )

        # --- Walk-Forward Companion (statistical inference) ---
        # When the primary mode is Monte Carlo, automatically run a walk-forward
        # expanding pass as a companion.  The model ranker will use WF p-values
        # for tier assignment and MC descriptors as screening gates.
        if mode == "monte_carlo":
            if not self.quiet:
                print(f"\n  Walk-Forward Companion (statistical inference):")

            # Build walk-forward commands for each model
            wf_model_cmds = []
            for model in self.candidate_models:
                cmd = [
                    "python",
                    "-m",
                    "algos.wfov.wfov_runner",
                    "--mode",
                    "walk_forward_expanding",
                    "--model_name",
                    model,
                    "--ticker",
                    self.ticker,
                    "--start_date",
                    start_date,
                    "--end_date",
                    end_date,
                    "--seed",
                    str(seed),
                    "--max_workers",
                    "1",
                    "--interval",
                    self.interval,
                    "--initial_train_days",
                    "504",
                    "--test_days",
                    "126",
                    "--step_days",
                    "126",
                    "--no-save-iterations",
                    "--no-plots",
                ]

                # Add model-specific params
                if model == "arima":
                    cmd.extend(
                        [
                            "--arima_threshold",
                            "0.0002",
                            "--arima_zscore",
                            "1.0",
                            "--arima_lookback",
                            "5",
                        ]
                    )

                wf_model_cmds.append((model, cmd))

            # Launch with bounded concurrency (max 3, same pattern as MC)
            wf_max_parallel = 3
            wf_passed = 0
            wf_failed = 0
            wf_failed_names = []
            wf_active_procs = []
            wf_pending = list(wf_model_cmds)

            while wf_pending or wf_active_procs:
                # Launch new processes up to max
                while wf_pending and len(wf_active_procs) < wf_max_parallel:
                    model, cmd = wf_pending.pop(0)
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=self._subprocess_env(),
                    )
                    wf_active_procs.append((model, proc))

                # Poll active processes for completion
                import time as _time

                wf_still_active = []
                for model, proc in wf_active_procs:
                    retcode = proc.poll()
                    if retcode is not None:
                        if retcode == 0:
                            wf_passed += 1
                        else:
                            wf_failed += 1
                            wf_failed_names.append(model)
                    else:
                        wf_still_active.append((model, proc))

                wf_active_procs = wf_still_active

                if wf_active_procs:
                    _time.sleep(0.5)

            if not self.quiet:
                wf_total = wf_passed + wf_failed
                if wf_failed == 0:
                    print(f"    Validated {wf_passed}/{wf_total} models (walk-forward)")
                else:
                    print(
                        f"    Validated {wf_passed}/{wf_total} models (walk-forward, "
                        f"{wf_failed} failed: {', '.join(wf_failed_names)})"
                    )

    def step3_rank_and_recommend(self, profile: str = "A", debug: bool = False) -> dict:
        """
        Step 3: Rank models and generate deployment recommendation.

        Args:
            profile: Ranking profile ('A', 'B', 'C')
            debug: Print debug information

        Returns:
            Dict with best model info and ranked dataframe
        """
        if not self.quiet:
            print(f"\n{'-' * 100}")
            print("STEP 3: Model Ranking & Selection")
            print("-" * 100)

        # Use profile-based ranker
        ranker = ModelRanker(profile=profile)

        # Find WFOV results from THIS workflow run only
        results_dir = Path("algos/wfov/results/summaries")
        all_summary_files = list(results_dir.glob(f"*_{self.ticker}_*.json"))

        # Filter by:
        # 1. Created after workflow started (with 2-minute buffer for clock skew)
        # 2. Model name matches one of our candidate models
        from datetime import timedelta

        cutoff = self.workflow_start_time - timedelta(minutes=2)

        expected_models = set(self.candidate_models)
        recent_files = []

        for f in all_summary_files:
            file_mtime = datetime.fromtimestamp(f.stat().st_mtime)
            created_after_start = file_mtime >= cutoff

            # Check if filename contains one of our expected models
            # Filename format: {mode}_{model}_{ticker}_{N}iter_*.json
            model_match = any(model in f.stem for model in expected_models)

            if created_after_start and model_match:
                recent_files.append(f)

        # Verify feature config hash consistency across results
        if recent_files and self._feature_hash:
            mixed_hashes = set()
            for f in recent_files:
                try:
                    with open(f, "r") as fh:
                        summary = json.load(fh)
                    file_hash = summary.get("metadata", {}).get("feature_config_hash")
                    if file_hash:
                        mixed_hashes.add(file_hash)
                except Exception:
                    pass
            if len(mixed_hashes) > 1 and not self.quiet:
                print(
                    f"  WARNING: Results generated with {len(mixed_hashes)} different feature configs detected"
                )
                print(f"  Hashes: {mixed_hashes}")
                print(f"  Current config hash: {self._feature_hash}")
                print(f"  Consider re-running WFOV to ensure consistency")

        if debug and not self.quiet:
            print(f"\nDEBUG: Result Collection")
            print(f"  Expected models: {list(expected_models)}")
            print(
                f"  Workflow started: {self.workflow_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"  Cutoff time: {cutoff.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Total files found: {len(all_summary_files)}")
            print(f"  Files after filtering: {len(recent_files)}")
            if self._feature_hash:
                print(f"  Feature config hash: {self._feature_hash}")

            if recent_files:
                print(f"  Matched files:")
                for f in recent_files:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    print(f"    - {f.name}")
                    print(f"      Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(
                        f"      Age: {(datetime.now() - mtime).total_seconds() / 60:.1f} minutes"
                    )

        if len(recent_files) < len(self.candidate_models) and not self.quiet:
            missing_count = len(self.candidate_models) - len(recent_files)
            print(
                f"⚠️  Only found {len(recent_files)} recent WFOV results (expected {len(self.candidate_models)})"
            )
            print(
                f"   Missing {missing_count} models - they may have failed validation"
            )

        if not recent_files:
            if not self.quiet:
                print(f"❌ No recent WFOV results found for {self.ticker}")
                print(f"   Check if WFOV validation completed successfully")
            return {"status": "no_results", "best_model": None, "_wfov_files": []}

        if not self.quiet:
            print(f"Found {len(recent_files)} WFOV results from current run")

        ranked_df = ranker.rank_models(recent_files, ticker=self.ticker)

        if ranked_df.empty:
            if not self.quiet:
                print(f"❌ No models could be ranked for {self.ticker}")
            return {"status": "no_models", "best_model": None, "_wfov_files": []}

        # Display results (not in quiet mode)
        if not self.quiet:
            print(f"\nRanked Models ({len(ranked_df)} total):")
            print("─" * 100)

            for idx, row in ranked_df.head(5).iterrows():
                m = row["metrics"]
                tier_symbols = {1: "🚀", 2: "⚠️", 3: "❌"}
                symbol = tier_symbols.get(row["tier"], "?")

                regime_info = ""
                if m["regime_dependent"]:
                    regime_info = (
                        f" | Regime: {m['regime_ratio']:.1f}x {m['best_regime']}"
                    )

                # Show feature count if available in the WFOV summary
                feat_info = ""
                fc = self._feature_configs.get(m["model_name"])
                if fc:
                    feat_info = f" | Feat: {len(fc.indicators)}"

                print(
                    f"{symbol} #{idx + 1}: {m['model_name']:20s} | "
                    f"Sharpe: {m['mean_sharpe']:5.2f} | "
                    f"p={m['p_value']:6.4f} | "
                    f"Score: {row['score']:5.2f}{feat_info}{regime_info}"
                )

            # Display comprehensive metrics for top 3 models
            print(f"\n{'─' * 100}")
            print("COMPREHENSIVE METRICS (Top 3 Models)")
            print("─" * 100)

            for idx, row in ranked_df.head(3).iterrows():
                m = row["metrics"]
                print(f"\n📊 {m['model_name'].upper()}")
                print(f"   {'─' * 50}")

                # Basic performance
                # NOTE: use `or 0` instead of relying on .get() defaults
                # because keys may exist with an explicit None value.
                print(
                    f"   Return:     {(m.get('mean_return') or 0):>8.2%}  |  Volatility:  {(m.get('mean_volatility') or 0):>8.2%}"
                )
                print(
                    f"   Sharpe:     {m['mean_sharpe']:>8.3f}  |  Hit Ratio:   {(m.get('hit_ratio') or 0):>8.2%}"
                )

                # Risk metrics
                max_dd = m.get("max_drawdown") or 0
                var_95 = m.get("var_95") or 0
                cvar_95 = m.get("cvar_95") or 0
                print(f"   Max DD:     {max_dd:>8.2%}  |  VaR (95%):   {var_95:>8.2%}")

                # Statistical
                psr = m.get("psr") or 0
                skewness = m.get("skewness") or 0
                kurtosis = m.get("kurtosis") or 3
                psr_indicator = "✓" if psr > 0.95 else "•" if psr > 0.85 else "✗"
                print(
                    f"   PSR:        {psr:>8.1%} {psr_indicator} |  Skewness:    {skewness:>8.3f}"
                )

                # Derived ratios
                sortino = m.get("sortino_ratio") or (m["mean_sharpe"] * 1.2)
                calmar = m.get("calmar_ratio") or (
                    (m.get("mean_return") or 0) / max_dd if max_dd > 0 else 0
                )
                print(f"   Sortino:    {sortino:>8.3f}  |  Calmar:      {calmar:>8.3f}")

                # Buy-and-hold benchmark
                bh_sh = m.get("bh_sharpe")
                exc_sh = m.get("excess_sharpe")
                ir = m.get("information_ratio")
                bh_str = f"{bh_sh:>8.3f}" if bh_sh is not None else "     N/A"
                exc_str = f"{exc_sh:>+8.3f}" if exc_sh is not None else "     N/A"
                ir_str = f"{ir:>8.3f}" if ir is not None else "     N/A"
                beat = "YES" if (exc_sh is not None and exc_sh > 0) else "NO"
                print(f"   B&H Sharpe: {bh_str}  |  Excess:      {exc_str}")
                print(f"   Info Ratio: {ir_str}  |  Beats B&H:   {'':>4}{beat}")

        # Get best model
        tier1_models = ranked_df[ranked_df["tier"] == 1]

        if not tier1_models.empty:
            best = tier1_models.iloc[0].to_dict()
            status = "deploy_recommended"
        else:
            tier2_models = ranked_df[ranked_df["tier"] == 2]
            if not tier2_models.empty:
                best = tier2_models.iloc[0].to_dict()
                status = "review_required"
            else:
                best = ranked_df.iloc[0].to_dict() if not ranked_df.empty else None
                status = "all_rejected"

        return {
            "status": status,
            "best_model": best,
            "ranked_df": ranked_df,
            "tier1_count": len(ranked_df[ranked_df["tier"] == 1]),
            "tier2_count": len(ranked_df[ranked_df["tier"] == 2]),
            "tier3_count": len(ranked_df[ranked_df["tier"] == 3]),
            "_wfov_files": recent_files,
        }

    def step4_show_deployment_command(self, best_model_info: dict):
        """Step 4: Show deployment recommendation for selected model."""
        if self.quiet:
            return

        print(f"\n{'-' * 100}")
        print("STEP 4: Deployment Recommendation")
        print("-" * 100)

        if best_model_info["status"] in ("no_models", "no_results"):
            print("No models available for deployment.")
            return

        if best_model_info["status"] == "all_rejected":
            print(
                "All models rejected. Consider alternative strategies or parameter tuning."
            )
            return

        best = best_model_info["best_model"]
        m = best["metrics"]

        # Core recommendation
        print(f"\n  Selected: {m['model_name'].upper()} ({best['tier_name']})")
        print(
            f"  Sharpe: {m['mean_sharpe']:.3f}  |  p-value: {m['p_value']:.4f}  |  Return: {(m.get('mean_return') or 0):.2%}  |  Max DD: {(m.get('max_drawdown') or 0):.2%}"
        )

        if m.get("regime_dependent"):
            print(
                f"  Regime risk: {m['regime_ratio']:.1f}x better in {m['best_regime']} -- monitor if market shifts"
            )

        # Single deploy command
        print(f"\n  Deploy:")
        print(
            f"    python algos/backtest_code/run_backtest_optimized.py --model_name {m['model_name']} --ticker {self.ticker} --start 2020-01-01 --end $(date +%Y-%m-%d)"
        )

    def _cleanup_wfov_artifacts(self, wfov_files: list):
        """Remove specific WFOV result files used during this workflow run."""
        for f in wfov_files:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    def run_full_workflow(self, **kwargs):
        """
        Execute complete workflow from training to deployment recommendation.

        Args:
            **kwargs: Parameters to pass through to WFOV validation and ranking
                validation_mode, profile, start_date, end_date, min_lookback_days, etc.
        """
        profile = kwargs.get("profile", "A")

        if not self.quiet:
            print(f"\n{'═' * 100}")
            print(f"AUTOMATED MODEL SELECTION WORKFLOW: {self.ticker}")
            print(f"Strategy: {ModelRanker.PROFILES[profile]['name']}")
            print(f"Philosophy: {ModelRanker.PROFILES[profile]['description']}")
            print("═" * 100)

        # Step 1: Train candidates
        self.step1_train_candidates()

        # Step 2: Validate with WFOV
        self.step2_validate_with_wfov(
            mode=kwargs.get("validation_mode", "monte_carlo"),
            seed=kwargs.get("seed", 42),
            start_date=kwargs.get("start_date", "2020-01-01"),
            end_date=kwargs.get("end_date"),
            min_lookback_days=kwargs.get("min_lookback_days", 365),
            max_lookback_days=kwargs.get("max_lookback_days", 1825),
            initial_train_days=kwargs.get("initial_train_days"),
            window_size=kwargs.get("window_size"),
            test_days=kwargs.get("test_days"),
            step_days=kwargs.get("step_days"),
        )

        # Step 3: Rank and select
        debug = kwargs.get("debug", False)
        result = self.step3_rank_and_recommend(profile=profile, debug=debug)

        # Clean up WFOV artifacts (summary JSONs were only needed for ranking)
        self._cleanup_wfov_artifacts(result.get("_wfov_files", []))

        # Step 4: Show deployment commands
        self.step4_show_deployment_command(result)

        return result


def _run_single_ticker_workflow(
    ticker: str,
    candidate_models: list,
    wfov_iterations: int,
    use_config: bool,
    config_path: str,
    preset: str,
    summary_only: bool,
    interval: str,
    no_plots: bool,
    from_store: bool,
    kwargs: dict,
) -> tuple:
    """
    Run a complete workflow for a single ticker. Used by both serial and parallel batch modes.

    Returns:
        (ticker, result_dict)
    """
    workflow = ModelSelectionWorkflow(
        ticker=ticker,
        candidate_models=candidate_models,
        wfov_iterations=wfov_iterations,
        parallel=True,
        use_config=use_config,
        config_path=config_path,
        preset=preset,
        quiet=summary_only,
        skip_training=kwargs.get("skip_training", False),
        skip_validation=kwargs.get("skip_validation", False),
        interval=interval,
        no_plots=no_plots or summary_only,
        from_store=from_store,
    )

    result = workflow.run_full_workflow(**kwargs)
    return (ticker, result)


def batch_workflow_all_tickers(
    tickers: list = None,
    tickers_json: str = None,
    append_tickers: list = None,
    candidate_models: list = None,
    wfov_iterations: int = 50,
    use_config: bool = True,
    config_path: str = "model_selection_config.yaml",
    preset: str = None,
    summary_only: bool = False,
    interval: str = "1d",
    no_plots: bool = False,
    from_store: bool = False,
    ticker_parallelism: int = 1,
    **kwargs,
):
    """
    Run workflow for all live trading tickers.

    Args:
        tickers: List of tickers (overrides config if provided)
        tickers_json: Path to JSON file with ticker keys (overrides config batch_tickers).
                      Ignored if tickers is explicitly provided.
        append_tickers: Extra tickers to append to the batch list (deduplicated).
                        Use for non-portfolio tickers like USDJPY, XAUUSD.
        candidate_models: Model types to test (overrides config)
        wfov_iterations: WFOV iterations per model
        use_config: Load ticker-specific models from config (default: True)
        config_path: Path to config file
        preset: Preset model list to use
        summary_only: Suppress intermediate output, only show final summary with all metrics
        interval: Data interval - '1d' (daily), '1wk' (weekly), '1mo' (monthly). Default: '1d'
        no_plots: Skip generating plots (still show output). Default: False
        from_store: Use parquet market data store only (no yfinance fallback).
        ticker_parallelism: Number of tickers to process in parallel (default: 1 = serial).
    """
    # Resolve tickers from the various sources (priority: --tickers > --tickers-json > config)
    ticker_source = None
    if tickers is not None:
        # Explicit --tickers takes highest priority
        ticker_source = "CLI --tickers"
        if tickers_json:
            if not summary_only:
                print(
                    f"⚠️  Both --tickers and --tickers-json provided; "
                    f"using --tickers (JSON ignored)"
                )
    elif tickers_json:
        # Load tickers from JSON file (keys only, weights ignored)
        tickers = load_tickers_from_json(tickers_json)
        ticker_source = f"JSON ({Path(tickers_json).name})"
        if not summary_only:
            print(
                f"✓ Loaded {len(tickers)} tickers from JSON: {Path(tickers_json).name}"
            )
    elif use_config:
        config = load_model_config(config_path)
        tickers = config.get("batch_tickers", ["NVDA", "AVGO", "8002.T", "III.L"])
        ticker_source = f"config ({Path(config_path).name})"
        if not summary_only:
            print(f"✓ Loaded batch tickers from config: {', '.join(tickers)}")
    else:
        tickers = ["NVDA", "AVGO", "8002.T", "III.L"]
        ticker_source = "hardcoded defaults"
        if not summary_only:
            print(f"ℹ️  Using hardcoded default tickers")

    # Append extra tickers (deduplicated, preserving order)
    if append_tickers:
        existing = set(tickers)
        added = []
        for t in append_tickers:
            if t not in existing:
                tickers.append(t)
                existing.add(t)
                added.append(t)
        if not summary_only:
            if added:
                print(f"✓ Appended {len(added)} extra ticker(s): {', '.join(added)}")
            else:
                print(
                    f"ℹ️  --append-tickers provided but all already in list (no new tickers added)"
                )

    if not summary_only:
        print(f"\n{'═' * 100}")
        print(f"BATCH MODEL SELECTION: {len(tickers)} TICKERS")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Data Interval: {interval}")
        if ticker_parallelism > 1:
            print(f"Ticker Parallelism: {ticker_parallelism} concurrent tickers")
        if use_config:
            print(f"Config: Loading models from {config_path}")
        if preset:
            print(f"Preset: Using '{preset}' model list for all tickers")
        print("═" * 100)
    else:
        print(f"\n{'═' * 100}")
        print(f"BATCH MODEL SELECTION (Summary Mode): {len(tickers)} TICKERS")
        print(f"Tickers: {', '.join(tickers)} | Interval: {interval}")
        if ticker_parallelism > 1:
            print(f"Ticker Parallelism: {ticker_parallelism}")
        print(f"Running... (intermediate output suppressed)")
        print("═" * 100)

    results = {}

    if ticker_parallelism > 1 and len(tickers) > 1:
        # Parallel ticker processing using ThreadPoolExecutor
        # (threads work fine here since each ticker spawns subprocesses for actual compute)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"\nProcessing {len(tickers)} tickers in parallel ({ticker_parallelism} concurrent)..."
        )

        with ThreadPoolExecutor(max_workers=ticker_parallelism) as executor:
            futures = {
                executor.submit(
                    _run_single_ticker_workflow,
                    ticker=ticker,
                    candidate_models=candidate_models,
                    wfov_iterations=wfov_iterations,
                    use_config=use_config,
                    config_path=config_path,
                    preset=preset,
                    summary_only=summary_only,
                    interval=interval,
                    no_plots=no_plots,
                    from_store=from_store,
                    kwargs=kwargs,
                ): ticker
                for ticker in tickers
            }

            completed = 0
            for future in as_completed(futures):
                ticker_name = futures[future]
                try:
                    ticker_name, result = future.result()
                    results[ticker_name] = result
                    completed += 1

                    if summary_only:
                        status_str = {
                            "deploy_recommended": "✓ (deploy)",
                            "review_required": "⚠ (review)",
                        }.get(result["status"], "✗ (reject)")
                        print(
                            f"  [{completed}/{len(tickers)}] {ticker_name}: {status_str}"
                        )
                except Exception as e:
                    completed += 1
                    results[ticker_name] = {
                        "status": "error",
                        "best_model": None,
                        "error": str(e),
                    }
                    print(
                        f"  [{completed}/{len(tickers)}] {ticker_name}: ✗ (error: {e})"
                    )
    else:
        # Serial ticker processing (original behavior)
        for i, ticker in enumerate(tickers, 1):
            if summary_only:
                print(
                    f"  [{i}/{len(tickers)}] Processing {ticker}...",
                    end=" ",
                    flush=True,
                )

            workflow = ModelSelectionWorkflow(
                ticker=ticker,
                candidate_models=candidate_models,
                wfov_iterations=wfov_iterations,
                parallel=True,
                use_config=use_config,
                config_path=config_path,
                preset=preset,
                quiet=summary_only,
                skip_training=kwargs.get("skip_training", False),
                skip_validation=kwargs.get("skip_validation", False),
                interval=interval,
                no_plots=no_plots or summary_only,
                from_store=from_store,
            )

            result = workflow.run_full_workflow(**kwargs)
            results[ticker] = result

            if summary_only:
                if result["status"] == "deploy_recommended":
                    print("✓ (deploy)")
                elif result["status"] == "review_required":
                    print("⚠ (review)")
                else:
                    print("✗ (reject)")
            else:
                print(f"\n{'─' * 100}\n")

    # Generate combined summary
    _print_batch_summary(results, summary_only)

    return results


def _print_batch_summary(results: dict, comprehensive: bool = False):
    """Print batch summary with optional comprehensive metrics."""
    print(f"\n{'═' * 100}")
    print("BATCH SUMMARY")
    print("═" * 100)

    # Show feature engineering config info
    try:
        from algos.common.feature_engine import FeatureConfig

        fc = FeatureConfig()
        if fc.indicators:
            print(f"Feature Config: {fc.describe()}")
    except (ImportError, Exception):
        pass

    # Quick summary table
    print(f"\n{'─' * 80}")
    print(
        f"{'TICKER':<12} {'STATUS':<10} {'MODEL':<20} {'SHARPE':>8} {'P-VALUE':>10} {'HIT RATE':>10}"
    )
    print(f"{'─' * 80}")

    for ticker, result in results.items():
        if result["status"] == "deploy_recommended":
            status = "🚀 DEPLOY"
            best = result["best_model"]["metrics"]
            model = best["model_name"].upper()
            sharpe = f"{best['mean_sharpe']:.2f}"
            pval = (
                f"{best.get('p_value', 0):.3f}"
                if best.get("p_value") is not None
                else "N/A"
            )
            hit = (
                f"{best.get('mean_hit_ratio', 0) * 100:.1f}%"
                if best.get("mean_hit_ratio")
                else "N/A"
            )
        elif result["status"] == "review_required":
            status = "⚠️  REVIEW"
            best = result["best_model"]["metrics"]
            model = best["model_name"].upper()
            sharpe = f"{best['mean_sharpe']:.2f}"
            pval = (
                f"{best.get('p_value', 0):.3f}"
                if best.get("p_value") is not None
                else "N/A"
            )
            hit = (
                f"{best.get('mean_hit_ratio', 0) * 100:.1f}%"
                if best.get("mean_hit_ratio")
                else "N/A"
            )
        else:
            status = "❌ REJECT"
            model = "-"
            sharpe = "-"
            pval = "-"
            hit = "-"

        print(f"{ticker:<12} {status:<10} {model:<20} {sharpe:>8} {pval:>10} {hit:>10}")

    print(f"{'─' * 80}")

    # Comprehensive metrics table (always show if available)
    if comprehensive or any(r.get("best_model") is not None for r in results.values()):
        print(f"\n{'═' * 100}")
        print("COMPREHENSIVE PERFORMANCE METRICS")
        print("═" * 100)

        # Header
        print(
            f"\n{'TICKER':<10} {'MODEL':<14} {'Sharpe':>7} {'B&H':>7} {'Excess':>7} "
            f"{'MaxDD':>7} {'Win%':>6} {'AnnRet':>8} {'AnnVol':>8} {'Sortino':>8} {'IR':>6}"
        )
        print(f"{'─' * 105}")

        for ticker, result in results.items():
            if result.get("best_model") and result["best_model"].get("metrics"):
                m = result["best_model"]["metrics"]
                model = m.get("model_name", "-")[:14].upper()

                # Extract metrics with safe defaults
                sharpe = m.get("mean_sharpe") or 0
                bh_sharpe = m.get("bh_sharpe")
                excess_sh = m.get("excess_sharpe")
                max_dd = m.get("max_drawdown") or 0
                hit_ratio = m.get("hit_ratio") or 0
                ann_return = m.get("mean_return") or 0
                ann_vol = m.get("mean_volatility") or 0
                sortino = m.get("sortino_ratio") or 0
                ir = m.get("information_ratio")

                # Format
                sharpe_str = f"{sharpe:7.2f}" if sharpe else "   N/A"
                bh_str = f"{bh_sharpe:7.2f}" if bh_sharpe is not None else "   N/A"
                excess_str = f"{excess_sh:+7.2f}" if excess_sh is not None else "   N/A"
                max_dd_str = f"{max_dd * 100:6.1f}%" if max_dd else "   N/A"
                hit_str = f"{hit_ratio * 100:5.1f}%" if hit_ratio else "  N/A"
                ret_str = f"{ann_return * 100:7.1f}%" if ann_return else "    N/A"
                vol_str = f"{ann_vol * 100:7.1f}%" if ann_vol else "    N/A"
                sortino_str = f"{sortino:8.2f}" if sortino else "     N/A"
                ir_str = f"{ir:6.2f}" if ir is not None else "  N/A"

                # Beat indicator
                beat = (
                    "+"
                    if (excess_sh is not None and excess_sh > 0)
                    else "-"
                    if (excess_sh is not None and excess_sh <= 0)
                    else "?"
                )

                print(
                    f"{ticker:<10} {model:<14} {sharpe_str} {bh_str} {excess_str} "
                    f"{max_dd_str} {hit_str} {ret_str} {vol_str} {sortino_str} {ir_str} {beat}"
                )
            else:
                print(
                    f"{ticker:<10} {'NO MODEL':<14} {'   N/A':>7} {'   N/A':>7} {'   N/A':>7} "
                    f"{'   N/A':>7} {'  N/A':>6} {'    N/A':>8} {'    N/A':>8} {'     N/A':>8} {'  N/A':>6}"
                )

        print(f"{'─' * 105}")

        # Risk metrics table
        print(
            f"\n{'TICKER':<10} {'MODEL':<12} {'Skew':>7} {'Kurt':>7} {'PSR':>7} "
            f"{'P-val':>8} {'RegimeDep':>10} {'Tier':>6}"
        )
        print(f"{'─' * 75}")

        for ticker, result in results.items():
            if result.get("best_model") and result["best_model"].get("metrics"):
                m = result["best_model"]["metrics"]
                model = m.get("model_name", "-")[:12].upper()

                skew = m.get("mean_skewness", m.get("skewness", 0))
                kurt = m.get("mean_kurtosis", m.get("kurtosis", 0))
                psr = m.get("psr", m.get("probabilistic_sharpe", 0))
                pval = m.get("p_value", 0)
                regime_dep = m.get("regime_dependency_ratio", 0)

                # Determine tier
                if result["status"] == "deploy_recommended":
                    tier = "TIER 1"
                elif result["status"] == "review_required":
                    tier = "TIER 2"
                else:
                    tier = "TIER 3"

                skew_str = f"{skew:7.2f}" if skew else "   N/A"
                kurt_str = f"{kurt:7.2f}" if kurt else "   N/A"
                psr_str = f"{psr:7.2f}" if psr else "   N/A"
                pval_str = f"{pval:8.3f}" if pval else "    N/A"
                regime_str = f"{regime_dep:10.2f}" if regime_dep else "       N/A"

                print(
                    f"{ticker:<10} {model:<12} {skew_str} {kurt_str} {psr_str} "
                    f"{pval_str} {regime_str} {tier:>6}"
                )
            else:
                print(
                    f"{ticker:<10} {'NO MODEL':<12} {'   N/A':>7} {'   N/A':>7} {'   N/A':>7} "
                    f"{'    N/A':>8} {'       N/A':>10} {'TIER 3':>6}"
                )

        print(f"{'─' * 75}")

    # Deployment summary
    tier1_count = sum(
        1 for r in results.values() if r["status"] == "deploy_recommended"
    )
    tier2_count = sum(1 for r in results.values() if r["status"] == "review_required")
    tier3_count = sum(
        1
        for r in results.values()
        if r["status"] in ("all_rejected", "no_results", "no_models", "error")
    )

    print(f"\n{'═' * 100}")
    print("DEPLOYMENT RECOMMENDATIONS")
    print("═" * 100)
    print(f"\n  🚀 TIER 1 (Deploy):  {tier1_count} tickers")
    print(f"  ⚠️  TIER 2 (Review):  {tier2_count} tickers")
    print(f"  ❌ TIER 3 (Reject):  {tier3_count} tickers")

    # Show deployment commands for Tier 1
    tier1_tickers = [
        t for t, r in results.items() if r["status"] == "deploy_recommended"
    ]
    if tier1_tickers:
        print(f"\n  Tier 1 Deployment Commands:")
        for ticker in tier1_tickers:
            result = results[ticker]
            model = result["best_model"]["metrics"]["model_name"]
            print(
                f"    python algos/backtest_code/run_backtest_optimized.py "
                f"--model_name {model} --ticker {ticker} --train_split 0.95"
            )

    print(f"\n{'═' * 100}")


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Model Selection Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow for single ticker
  python scripts/model_selection_workflow.py --ticker NVDA

  # Run for all live tickers
  python scripts/model_selection_workflow.py --batch

  # Batch mode with summary only (suppress intermediate output)
  python scripts/model_selection_workflow.py --batch --summary-only

  # Batch with specific tickers and summary only
  python scripts/model_selection_workflow.py --tickers NVDA AVGO SPY --summary-only

  # Custom model list
  python scripts/model_selection_workflow.py \\
      --ticker SPY \\
      --models lstm svm_optimized xgb_optimized \\
      --iterations 100

  # Quick mode (fewer iterations for fast screening)
  python scripts/model_selection_workflow.py --ticker NVDA --quick

  # Skip training/validation (use existing results)
  python scripts/model_selection_workflow.py --batch --skip-training --skip-validation --summary-only

  # Weekly timeframe backtest (instead of daily)
  python scripts/model_selection_workflow.py --ticker SPY --interval 1wk

  # Batch with weekly data
  python scripts/model_selection_workflow.py --batch --interval 1wk --summary-only

  # Force parquet store only (no yfinance fallback)
  python scripts/model_selection_workflow.py --ticker NVDA --from-store

  # Use a portfolio weights JSON as the ticker source (replaces config batch_tickers)
  python scripts/model_selection_workflow.py \\
      --tickers-json algos/backtest_code/data/mc_1_weights.json --summary-only

  # JSON tickers + append non-portfolio tickers (e.g., FX pairs)
  python scripts/model_selection_workflow.py \\
      --tickers-json algos/backtest_code/data/hrp_weights.json \\
      --append-tickers USDJPY XAUUSD --summary-only

  # Append extra tickers to the config batch list
  python scripts/model_selection_workflow.py --batch --append-tickers USDJPY --summary-only
        """,
    )

    parser.add_argument("--ticker", type=str, help="Single ticker to analyze")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run for all live tickers (uses batch_tickers from config)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="+",
        help="Specific tickers for batch mode (overrides config batch_tickers)",
    )
    parser.add_argument(
        "--tickers-json",
        type=str,
        help="Path to a JSON file whose keys are ticker symbols (e.g., portfolio weights files "
        "like mc_1_weights.json). Extracts ticker keys only (weights are ignored). "
        "Overrides batch_tickers from YAML config. Ignored if --tickers is also provided.",
    )
    parser.add_argument(
        "--append-tickers",
        type=str,
        nargs="+",
        help="Extra tickers to append to the batch list from any source (JSON, YAML config, "
        "or --tickers). Use for non-portfolio tickers like USDJPY, XAUUSD. Deduplicated.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Model types to test (overrides config file and presets)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "comprehensive"],
        help="Use preset model list from config: 'quick' (2 models, fast) or 'comprehensive' (12+ models, thorough)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="WFOV iterations per model (default: 50)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 20 iterations + quick_models preset (fastest)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Don't use config file (use hardcoded defaults or --models)",
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help="Use parquet market data store only (disable network fallback)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="model_selection_config.yaml",
        help="Path to config file (default: model_selection_config.yaml)",
    )

    # Ranking profile
    parser.add_argument(
        "--profile",
        type=str,
        choices=["A", "B", "C"],
        default="A",
        help="Ranking profile: A (max returns), B (risk-adjusted), C (institutional)",
    )

    # WFOV validation mode and parameters
    parser.add_argument(
        "--validation-mode",
        type=str,
        choices=["monte_carlo", "walk_forward_expanding", "walk_forward_rolling"],
        default="monte_carlo",
        help="WFOV validation mode (default: monte_carlo)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="WFOV start date (default: 2020-01-01)",
    )
    parser.add_argument("--end-date", type=str, help="WFOV end date (default: today)")
    parser.add_argument(
        "--lookback-days",
        type=int,
        help="Lookback period in calendar days from today. Overrides --start-date and --end-date. "
        "E.g., --lookback-days 1825 sets start to ~5 years ago and end to today.",
    )
    parser.add_argument(
        "--min-lookback",
        type=int,
        default=365,
        help="Min lookback days for Monte Carlo (default: 365)",
    )
    parser.add_argument(
        "--max-lookback",
        type=int,
        default=1825,
        help="Max lookback days for Monte Carlo (default: 1825)",
    )
    parser.add_argument(
        "--initial-train-days",
        type=int,
        help="Initial training days for walk-forward expanding",
    )
    parser.add_argument(
        "--window-size", type=int, help="Window size for walk-forward rolling"
    )
    parser.add_argument(
        "--test-days", type=int, help="Test period size for walk-forward modes"
    )
    parser.add_argument(
        "--step-days", type=int, help="Step size for walk-forward modes"
    )

    # Data interval
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval: '1d' (daily), '1wk' (weekly), '1mo' (monthly). Default: 1d",
    )

    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed debug information for troubleshooting",
    )

    # Summary-only mode for batch
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Batch mode: suppress logs/images, only show final summary with all metrics",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots but still show output (less quiet than --summary-only)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step (use existing models)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip WFOV validation step (use existing validation results)",
    )
    parser.add_argument(
        "--ticker-parallelism",
        type=int,
        default=1,
        help="Number of tickers to process in parallel in batch mode (default: 1 = serial). "
        "Recommended: 2-4 for CPU-bound workloads. Be careful with high values as "
        "each ticker spawns its own model training and WFOV subprocesses.",
    )
    # Legacy flags -- these are now always-on (workflow never writes artifacts).
    # Accepted for backward compatibility so existing scripts don't break.
    parser.add_argument(
        "--skip-model-save",
        action="store_true",
        default=True,
        help="(No-op, always enabled) Workflow never saves model/scaler files.",
    )
    parser.add_argument(
        "--no-save-intermediates",
        action="store_true",
        default=True,
        help="(No-op, always enabled) Workflow never saves intermediate CSVs.",
    )
    parser.add_argument(
        "--no-save-iterations",
        action="store_true",
        default=True,
        help="(No-op, always enabled) Workflow never saves WFOV iteration CSVs.",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove old artifacts (WFOV results, model dumps, intermediate CSVs) older than --cleanup-days",
    )
    parser.add_argument(
        "--cleanup-days",
        type=int,
        default=30,
        help="Remove artifacts older than this many days (default: 30). Used with --cleanup.",
    )
    parser.add_argument(
        "--cleanup-dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting (use with --cleanup)",
    )

    args = parser.parse_args()

    # Validate walk-forward parameters
    if args.validation_mode == "walk_forward_expanding":
        if not all([args.initial_train_days, args.test_days, args.step_days]):
            parser.error(
                f"\n❌ Walk-Forward Expanding mode requires:\n"
                f"   --initial-train-days <days>  (e.g., 1260 for 5 years)\n"
                f"   --test-days <days>           (e.g., 252 for 1 year)\n"
                f"   --step-days <days>           (e.g., 126 for 6 months)\n"
                f"\nExample:\n"
                f"  python scripts/model_selection_workflow.py \\\n"
                f"      --ticker {args.ticker or 'SPY'} \\\n"
                f"      --validation-mode walk_forward_expanding \\\n"
                f"      --initial-train-days 1260 --test-days 252 --step-days 126\n"
            )
    elif args.validation_mode == "walk_forward_rolling":
        if not all([args.window_size, args.test_days, args.step_days]):
            parser.error(
                f"\n❌ Walk-Forward Rolling mode requires:\n"
                f"   --window-size <days>  (e.g., 1260 for 5-year window)\n"
                f"   --test-days <days>    (e.g., 252 for 1 year)\n"
                f"   --step-days <days>    (e.g., 126 for 6 months)\n"
                f"\nExample:\n"
                f"  python scripts/model_selection_workflow.py \\\n"
                f"      --ticker {args.ticker or 'SPY'} \\\n"
                f"      --validation-mode walk_forward_rolling \\\n"
                f"      --window-size 1260 --test-days 252 --step-days 126\n"
            )

    # Determine iterations and preset
    if args.quick:
        iterations = 20
        preset = (
            args.preset or "quick"
        )  # --quick implies quick preset unless overridden
    else:
        iterations = args.iterations
        preset = args.preset  # Use specified preset or None

    # Determine config usage
    use_config = not args.no_config

    # Setup logging to file
    log_dir = Path("workflow_results")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.ticker:
        log_file = log_dir / f"{args.ticker}_{timestamp}.txt"
    elif args.tickers:
        log_file = log_dir / f"batch_{'_'.join(args.tickers[:3])}_{timestamp}.txt"
    elif args.tickers_json:
        json_stem = Path(args.tickers_json).stem
        log_file = log_dir / f"batch_{json_stem}_{timestamp}.txt"
    else:
        log_file = log_dir / f"batch_{timestamp}.txt"

    # Compute dates from --lookback-days if provided (overrides --start-date / --end-date)
    if args.lookback_days:
        from datetime import timedelta

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=args.lookback_days)
        args.end_date = end_dt.strftime("%Y-%m-%d")
        args.start_date = start_dt.strftime("%Y-%m-%d")

    # Build kwargs for WFOV and ranking parameters
    wfov_kwargs = {
        "validation_mode": args.validation_mode,
        "profile": args.profile,
        "seed": args.seed,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "min_lookback_days": args.min_lookback,
        "max_lookback_days": args.max_lookback,
        "initial_train_days": args.initial_train_days,
        "window_size": args.window_size,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "debug": args.debug,
        "skip_training": args.skip_training,
        "skip_validation": args.skip_validation,
    }

    # Handle cleanup mode
    if args.cleanup:
        cleanup_old_artifacts(
            max_age_days=args.cleanup_days,
            dry_run=args.cleanup_dry_run,
        )
        if not (args.batch or args.tickers or args.tickers_json or args.ticker):
            sys.exit(0)  # Cleanup only, no workflow to run

    # Execute with dual logging (console + file)
    with DualLogger(log_file) as logger:
        print(f"📝 Logging to: {log_file}")
        print()

        if args.batch or args.tickers or args.tickers_json:
            # Batch mode: multiple tickers
            batch_workflow_all_tickers(
                tickers=args.tickers,  # None if --batch without --tickers (loads from config/JSON)
                tickers_json=args.tickers_json,
                append_tickers=args.append_tickers,
                candidate_models=args.models,
                wfov_iterations=iterations,
                use_config=use_config,
                config_path=args.config_path,
                preset=preset,
                summary_only=args.summary_only,
                interval=args.interval,
                no_plots=args.no_plots,
                from_store=args.from_store,
                ticker_parallelism=args.ticker_parallelism,
                **wfov_kwargs,
            )

        elif args.ticker:
            # Single ticker workflow
            workflow = ModelSelectionWorkflow(
                ticker=args.ticker,
                candidate_models=args.models,
                wfov_iterations=iterations,
                parallel=True,
                use_config=use_config,
                config_path=args.config_path,
                preset=preset,
                interval=args.interval,
                no_plots=args.no_plots,
                from_store=args.from_store,
            )

            workflow.run_full_workflow(**wfov_kwargs)

        else:
            parser.error("Must specify --ticker, --batch, --tickers, or --tickers-json")

    # Print log file location after completion
    print(f"\n{'═' * 100}")
    print(f"📝 Full output saved to: {log_file}")
    print(f"   View anytime: cat {log_file}")
    print("═" * 100)


if __name__ == "__main__":
    main()
