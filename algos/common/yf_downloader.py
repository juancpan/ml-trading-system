"""
Centralized resilient yfinance download utility.

Provides robust data downloading with:
- Exponential backoff with jitter on any error
- Failed-ticker detection and selective retry (only re-downloads tickers that failed)
- SQLite cache corruption recovery (clears yfinance's internal cache on OperationalError)
- Configurable retries, delays, and batch sizes
- Detailed logging of every retry attempt and final outcome

All yfinance data downloading across the project should go through this module.

Usage:
    from algos.common.yf_downloader import resilient_download, resilient_download_single

    # Single ticker
    df = resilient_download_single("SPY", start="2020-01-01", end="2024-01-01")

    # Multiple tickers (batched, with failed-ticker retry)
    df = resilient_download(["SPY", "AAPL", "MSFT"], start="2020-01-01", end="2024-01-01")
"""

import yfinance as yf
import pandas as pd
import logging
import time
import random
import re
import io
import sys
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# yfinance/stderr capture is process-global; protect single-call sections.
_YF_SINGLE_CALL_LOCK = threading.Lock()


# =============================================================================
# CONFIGURATION
# =============================================================================

DOWNLOAD_CONFIG = {
    # Retry settings
    "max_retries": 5,  # Total attempts per ticker/batch
    "base_delay": 3.0,  # Initial retry delay (seconds)
    "backoff_multiplier": 2.0,  # Exponential backoff factor
    "max_delay": 120.0,  # Cap on retry delay (seconds)
    "jitter_fraction": 0.5,  # Random jitter as fraction of delay (0.0-1.0)
    # Batching settings (for multi-ticker downloads)
    "batch_size": 25,  # Tickers per yf.download() call
    "delay_between_batches": 4.0,  # Base delay between batches (seconds)
    "batch_jitter": 2.0,  # Random jitter added to batch delay
    # Failure detection
    "partial_failure_max_retries": 3,  # Extra retries for tickers that failed within a batch
}

# Patterns in yfinance stderr/error output that indicate retriable failures
_RETRIABLE_PATTERNS = [
    r"429",  # Rate limited
    r"too many requests",
    r"rate",
    r"timeout",
    r"timed out",
    r"connection",
    r"connectionerror",
    r"connectionreseterror",
    r"remotedisconnected",
    r"urlerror",
    r"sslerror",
    r"brokenpipeerror",
    r"operationalerror",  # SQLite cache corruption
    r"unable to open database",
    r"database is locked",
    r"nonetype.*not subscriptable",  # yfinance internal failure on bad response
    r"json",  # JSON decode errors from bad response
    r"chunkedencodingerror",
    r"incompleteread",
    r"protocolerror",
    r"too many open files",  # fd exhaustion from threaded downloads
    r"oserror",
    r"errno 24",
]

# Patterns that indicate yfinance's SQLite cache is corrupted and must be cleared
_CACHE_CORRUPTION_PATTERNS = re.compile(
    r"operationalerror|unable to open database|database is locked|"
    r"too many open files|errno 24",
    re.IGNORECASE,
)

_RETRIABLE_RE = re.compile("|".join(_RETRIABLE_PATTERNS), re.IGNORECASE)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _compute_delay(
    attempt: int,
    base_delay: float = None,
    backoff: float = None,
    max_delay: float = None,
    jitter_fraction: float = None,
) -> float:
    """Compute retry delay with exponential backoff and jitter."""
    base_delay = base_delay or DOWNLOAD_CONFIG["base_delay"]
    backoff = backoff or DOWNLOAD_CONFIG["backoff_multiplier"]
    max_delay = max_delay or DOWNLOAD_CONFIG["max_delay"]
    jitter_fraction = (
        jitter_fraction
        if jitter_fraction is not None
        else DOWNLOAD_CONFIG["jitter_fraction"]
    )

    delay = min(base_delay * (backoff**attempt), max_delay)
    jitter = delay * jitter_fraction * random.random()
    return delay + jitter


def _is_retriable(error: Exception) -> bool:
    """Check if an exception represents a transient/retriable failure."""
    error_str = str(error)
    error_type = type(error).__name__
    combined = f"{error_type}: {error_str}"
    return bool(_RETRIABLE_RE.search(combined))


def _clear_yfinance_cache() -> None:
    """
    Clear yfinance's internal SQLite cache to recover from OperationalError.

    yfinance uses an internal cache directory (usually ~/.cache/py-yfinance/)
    that can get corrupted on flaky networks (partial writes, locked DB, etc).
    This also attempts to disable yfinance's cache entirely for the remainder
    of the session so it won't re-corrupt on subsequent calls.
    """
    cache_dirs = [
        Path.home() / ".cache" / "py-yfinance",
        Path.home() / ".cache" / "yfinance",
    ]
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleared corrupted yfinance cache: {cache_dir}")
            except Exception as e:
                logger.warning(f"Could not clear yfinance cache {cache_dir}: {e}")

    # Disable yfinance's internal cache for the rest of this session to prevent
    # re-corruption. yfinance uses requests_cache or its own CachingSession.
    try:
        yf.set_tz_cache_location(str(Path.home() / ".cache" / "py-yfinance-tz"))
    except (AttributeError, Exception):
        pass

    try:
        # Disable the cache entirely if yfinance exposes the mechanism
        import yfinance.cache as yf_cache

        if hasattr(yf_cache, "set_tz_cache_location"):
            # Point to a fresh temp dir
            import tempfile

            tmp_cache = Path(tempfile.mkdtemp(prefix="yf_cache_"))
            yf_cache.set_tz_cache_location(str(tmp_cache))
            logger.info(f"Redirected yfinance tz cache to temp dir: {tmp_cache}")
    except (ImportError, Exception):
        pass


def _has_cache_corruption(text: str) -> bool:
    """Check if error text contains patterns indicating yfinance cache corruption."""
    return bool(_CACHE_CORRUPTION_PATTERNS.search(text))


@contextmanager
def _capture_stderr():
    """
    Context manager to capture stderr output from yfinance.

    yfinance prints download failures to stderr rather than raising exceptions.
    We capture this to detect which tickers failed within a batch.
    """
    old_stderr = sys.stderr
    sys.stderr = captured = io.StringIO()
    try:
        yield captured
    finally:
        sys.stderr = old_stderr


def _parse_failed_tickers(stderr_text: str, requested_tickers: list) -> list:
    """
    Parse yfinance stderr output to identify which tickers failed.

    yfinance reports failures like:
        [***] 2 Failed downloads:
        ['AAPL', 'MCO']: OperationalError('unable to open database file')
        ['AXP', 'V']: TypeError("'NoneType' object is not subscriptable")

    Also handles single-ticker format:
        [AAPL]: OperationalError('unable to open database file')

    Args:
        stderr_text: Captured stderr from yfinance download
        requested_tickers: List of tickers that were requested

    Returns:
        List of ticker symbols that failed
    """
    if not stderr_text.strip():
        return []

    failed = set()

    # Pattern 1: List format ['TICKER1', 'TICKER2']: ErrorType(...)
    list_pattern = re.compile(r"\['([^]]+)'\]\s*:\s*\w+")
    for match in list_pattern.finditer(stderr_text):
        tickers_str = match.group(1)
        for ticker in re.findall(r"'([^']+)'", f"'{tickers_str}'"):
            ticker = ticker.strip()
            if ticker in requested_tickers:
                failed.add(ticker)

    # Pattern 2: Also catch individual bracket format [TICKER]: Error
    bracket_pattern = re.compile(r"\[([A-Z0-9.\-=^]+)\]\s*:\s*\w+")
    for match in bracket_pattern.finditer(stderr_text):
        ticker = match.group(1).strip()
        if ticker in requested_tickers:
            failed.add(ticker)

    # Pattern 3: Catch "Failed download" lines and extract tickers from them
    # Sometimes yfinance formats: "- AAPL: No data found"
    dash_pattern = re.compile(r"^[\s-]+([A-Z0-9.\-=^]+)\s*:", re.MULTILINE)
    for match in dash_pattern.finditer(stderr_text):
        ticker = match.group(1).strip()
        if ticker in requested_tickers:
            failed.add(ticker)

    return list(failed)


def _detect_missing_tickers(data: pd.DataFrame, requested_tickers: list) -> list:
    """
    Detect tickers that are completely missing from the downloaded data.

    When yfinance returns partial results, some tickers may have zero rows
    or be entirely absent from the MultiIndex columns.

    Args:
        data: Downloaded DataFrame (possibly MultiIndex columns)
        requested_tickers: List of tickers that were requested

    Returns:
        List of ticker symbols that are missing or have no data
    """
    if data is None or data.empty:
        return list(requested_tickers)

    missing = []
    columns = data.columns

    if isinstance(columns, pd.MultiIndex):
        # Multi-ticker download: columns are (Price, Ticker)
        # Check both level 0 and level 1 for ticker names
        available_tickers = set()
        for level_idx in range(columns.nlevels):
            available_tickers.update(columns.get_level_values(level_idx))

        for ticker in requested_tickers:
            if ticker not in available_tickers:
                missing.append(ticker)
            else:
                # Ticker exists in columns but might have all-NaN data
                try:
                    ticker_data = data.xs(ticker, axis=1, level=1, drop_level=False)
                    if ticker_data.dropna(how="all").empty:
                        missing.append(ticker)
                except (KeyError, TypeError):
                    missing.append(ticker)
    else:
        # Single-ticker download: columns are just price names
        # If we got data back, the single ticker succeeded
        if len(requested_tickers) == 1 and not data.empty:
            return []
        # For single ticker with empty data
        if data.empty:
            return list(requested_tickers)

    return missing


# =============================================================================
# PUBLIC API
# =============================================================================


def resilient_download_single(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    max_retries: int = None,
    progress: bool = False,
    ignore_tz: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Download data for a single ticker with robust retry logic.

    Retries on transient errors (network timeouts, rate limits, SQLite
    corruption) with exponential backoff and jitter.

    Args:
        ticker: Ticker symbol (e.g., 'SPY', 'BTC-USD')
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        interval: Data interval (default '1d')
        auto_adjust: Whether to auto-adjust prices (default True)
        max_retries: Override default max retries
        progress: Whether to show yfinance progress bar
        ignore_tz: Whether to ignore timezone when combining data from different
                   exchanges. Default True (strip tz info for uniform naive datetimes).

    Returns:
        DataFrame with downloaded data, or None if all retries exhausted
    """
    max_retries = max_retries or DOWNLOAD_CONFIG["max_retries"]
    cache_corruption_seen = False

    for attempt in range(max_retries):
        # Clear cache before every retry if corruption was previously detected
        if cache_corruption_seen and attempt > 0:
            _clear_yfinance_cache()

        try:
            # yfinance's stderr output is noisy and global; for single-ticker
            # calls we rely on return value + exceptions and force non-threaded
            # download to reduce fd/thread pressure under flaky networks.
            with _YF_SINGLE_CALL_LOCK:
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=progress,
                    auto_adjust=auto_adjust,
                    ignore_tz=ignore_tz,
                    threads=False,
                )

            if data is not None and not data.empty:
                if attempt > 0:
                    logger.info(
                        f"Successfully downloaded {ticker} on attempt {attempt + 1}"
                    )
                return data

            # Empty data - could be transient
            if attempt < max_retries - 1:
                delay = _compute_delay(attempt)
                logger.warning(
                    f"Empty response for {ticker} (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        except Exception as e:
            error_str = str(e).lower()

            # Detect cache corruption from exception
            if _has_cache_corruption(error_str) and not cache_corruption_seen:
                _clear_yfinance_cache()
                cache_corruption_seen = True

            if attempt < max_retries - 1:
                delay = _compute_delay(attempt)
                logger.warning(
                    f"Error downloading {ticker} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Failed to download {ticker} after {max_retries} attempts. "
                    f"Last error: {e}"
                )

    return None


def resilient_download(
    tickers: list,
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = False,
    threads: bool = True,
    max_retries: int = None,
    batch_size: int = None,
    progress: bool = False,
    ignore_tz: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Download data for multiple tickers with batching, retry, and failed-ticker recovery.

    This is the main entry point for multi-ticker downloads. It:
    1. Splits tickers into batches to avoid rate limits
    2. Downloads each batch with retry on failure
    3. Detects which individual tickers failed within a batch (via stderr parsing)
       -- including when ALL tickers in a retry fail (empty response + stderr errors)
    4. Clears yfinance's SQLite cache before EVERY retry when OperationalError is seen
       (the cache re-corrupts on each download attempt on flaky networks)
    5. Disables threading when 'Too many open files' is detected
    6. Retries only the failed tickers individually with exponential backoff (Phase 2)
    7. Merges all results into a single DataFrame

    Args:
        tickers: List of ticker symbols
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        interval: Data interval (default '1d')
        auto_adjust: Whether to auto-adjust prices (default False for backward compat)
        threads: Whether to use threading in yf.download
        max_retries: Override default max retries for batch-level retry
        batch_size: Override default batch size
        progress: Whether to show yfinance progress bar
        ignore_tz: Whether to ignore timezone when combining data from different
                   exchanges. Default True (strip tz info for uniform naive datetimes).

    Returns:
        DataFrame with downloaded data (MultiIndex columns if >1 ticker),
        or None if complete failure
    """
    if not tickers:
        logger.warning("No tickers provided to resilient_download")
        return pd.DataFrame()

    max_retries = max_retries or DOWNLOAD_CONFIG["max_retries"]
    batch_size = batch_size or DOWNLOAD_CONFIG["batch_size"]
    partial_retries = DOWNLOAD_CONFIG["partial_failure_max_retries"]
    use_threads = threads

    # For a single ticker, delegate to single-ticker function
    if len(tickers) == 1:
        return resilient_download_single(
            tickers[0],
            start,
            end,
            interval,
            auto_adjust=auto_adjust,
            max_retries=max_retries,
            progress=progress,
            ignore_tz=ignore_tz,
        )

    # Track whether we've seen cache corruption in this session so we know to
    # proactively clear before each retry (not just once).
    cache_corruption_seen = False

    all_batch_results = []
    all_failed_tickers = []

    # Phase 1: Batch downloads
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    for batch_idx in range(0, len(tickers), batch_size):
        batch = tickers[batch_idx : batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1

        logger.info(
            f"Batch {batch_num}/{total_batches}: Downloading {len(batch)} tickers..."
        )

        batch_failed = list(batch)  # Assume all failed until proven otherwise

        for attempt in range(max_retries):
            # Proactively clear cache before every retry if we've ever seen
            # corruption. The cache re-corrupts on each yf.download() call on
            # flaky networks, so clearing once is not enough.
            if cache_corruption_seen and attempt > 0:
                _clear_yfinance_cache()

            try:
                with _capture_stderr() as captured:
                    batch_data = yf.download(
                        batch_failed,
                        start=start,
                        end=end,
                        interval=interval,
                        threads=use_threads,
                        auto_adjust=auto_adjust,
                        progress=progress,
                        ignore_tz=ignore_tz,
                    )

                stderr_text = captured.getvalue()

                # Always check stderr for errors, even when data is empty.
                # When ALL tickers in a retry fail, yfinance returns an empty
                # DataFrame but still writes the errors to stderr.
                stderr_failed = (
                    _parse_failed_tickers(stderr_text, batch_failed)
                    if stderr_text.strip()
                    else []
                )

                # Detect cache corruption in stderr
                if stderr_text.strip() and _has_cache_corruption(stderr_text):
                    if not cache_corruption_seen:
                        logger.warning(
                            f"Batch {batch_num}: Detected yfinance cache corruption, "
                            f"will clear cache before each retry"
                        )
                        _clear_yfinance_cache()
                        cache_corruption_seen = True

                # Detect 'Too many open files' and disable threading
                if stderr_text.strip() and re.search(
                    r"too many open files|errno 24", stderr_text, re.IGNORECASE
                ):
                    if use_threads:
                        logger.warning(
                            f"Batch {batch_num}: 'Too many open files' detected, "
                            f"disabling threaded downloads for remainder of session"
                        )
                        use_threads = False

                has_data = batch_data is not None and not batch_data.empty
                data_missing = (
                    _detect_missing_tickers(batch_data, batch_failed)
                    if has_data
                    else list(batch_failed)
                )
                newly_failed = list(set(stderr_failed) | set(data_missing))
                succeeded = [t for t in batch_failed if t not in newly_failed]

                if has_data and succeeded:
                    # We got data for at least some tickers - save it.
                    # Strip out columns for failed tickers to avoid NaN pollution.
                    if newly_failed and isinstance(batch_data.columns, pd.MultiIndex):
                        # Keep only columns for succeeded tickers
                        succeeded_set = set(succeeded)
                        keep_cols = [
                            col
                            for col in batch_data.columns
                            if col[1] in succeeded_set  # (Price, Ticker) format
                        ]
                        if keep_cols:
                            clean_data = batch_data[keep_cols].copy()
                            all_batch_results.append(clean_data)
                        else:
                            all_batch_results.append(batch_data)
                    else:
                        all_batch_results.append(batch_data)

                    logger.info(
                        f"Batch {batch_num}: {len(succeeded)}/{len(batch_failed)} tickers succeeded"
                    )

                if newly_failed:
                    if attempt < max_retries - 1:
                        delay = _compute_delay(attempt)
                        logger.warning(
                            f"Batch {batch_num}: {len(newly_failed)}/{len(batch_failed)} tickers failed "
                            f"(attempt {attempt + 1}/{max_retries}). "
                            f"Retrying {len(newly_failed)} failed tickers in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        batch_failed = newly_failed
                        continue
                    else:
                        logger.error(
                            f"Batch {batch_num}: {len(newly_failed)} tickers still failing after "
                            f"{max_retries} attempts: {newly_failed}"
                        )
                        all_failed_tickers.extend(newly_failed)
                        break
                else:
                    # All tickers succeeded
                    break

            except Exception as e:
                error_str = str(e).lower()
                if _has_cache_corruption(error_str):
                    _clear_yfinance_cache()
                    cache_corruption_seen = True

                if "too many open files" in error_str or "errno 24" in error_str:
                    use_threads = False
                    logger.warning(
                        f"Batch {batch_num}: 'Too many open files' exception, "
                        f"disabling threaded downloads"
                    )

                if attempt < max_retries - 1:
                    delay = _compute_delay(attempt)
                    logger.warning(
                        f"Batch {batch_num}: Exception on attempt {attempt + 1}/{max_retries}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Batch {batch_num}: Failed after {max_retries} attempts. Last error: {e}"
                    )
                    all_failed_tickers.extend(batch_failed)

        # Delay between batches (except last)
        if batch_idx + batch_size < len(tickers):
            delay = DOWNLOAD_CONFIG["delay_between_batches"] + random.uniform(
                0, DOWNLOAD_CONFIG["batch_jitter"]
            )
            logger.info(f"  Waiting {delay:.1f}s before next batch...")
            time.sleep(delay)

    # Phase 2: Individual retry for persistently failed tickers
    if all_failed_tickers:
        # Deduplicate (a ticker could appear from multiple batch retries)
        unique_failed = list(dict.fromkeys(all_failed_tickers))
        logger.info(
            f"Phase 2: Retrying {len(unique_failed)} individually failed tickers "
            f"one at a time..."
        )

        # Clear cache one more time before Phase 2 if corruption was seen
        if cache_corruption_seen:
            _clear_yfinance_cache()

        individual_results = []
        still_failed = []

        for ticker in unique_failed:
            # Clear cache before each individual ticker if corruption was seen
            if cache_corruption_seen:
                _clear_yfinance_cache()

            single_data = resilient_download_single(
                ticker,
                start,
                end,
                interval,
                auto_adjust=auto_adjust,
                max_retries=partial_retries,
                progress=progress,
                ignore_tz=ignore_tz,
            )
            if single_data is not None and not single_data.empty:
                # Wrap single-ticker data in MultiIndex to match batch format
                if not isinstance(single_data.columns, pd.MultiIndex):
                    single_data.columns = pd.MultiIndex.from_product(
                        [single_data.columns, [ticker]]
                    )
                individual_results.append(single_data)
                logger.info(f"  Recovered {ticker} via individual download")
            else:
                still_failed.append(ticker)

            # Small delay between individual downloads to avoid rate limiting
            time.sleep(random.uniform(1.0, 2.5))

        if individual_results:
            all_batch_results.extend(individual_results)

        if still_failed:
            logger.error(
                f"Final failures after all retries: {len(still_failed)} tickers: {still_failed}"
            )

    # Phase 3: Merge all results
    if not all_batch_results:
        logger.error("No data downloaded for any ticker after all retries")
        return pd.DataFrame()

    if len(all_batch_results) == 1:
        return all_batch_results[0]

    combined = all_batch_results[0]
    for df in all_batch_results[1:]:
        combined = combined.join(df, how="outer")

    logger.info(
        f"Download complete: {len(tickers)} requested, "
        f"combined data shape: {combined.shape}"
    )

    return combined
