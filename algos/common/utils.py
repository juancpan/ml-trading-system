# algos/common/utils.py

import os
import sys
import re
import datetime as dt
from pylab import mpl  # Import mpl here for font family setting
from algos.common.config import LOGS_DIR  # Import LOGS_DIR from config

# Set font family for plots (if not set globally in config already)
mpl.rcParams["font.family"] = "serif"

# Global flag: when True, all RedirectStdoutToFile instances write to /dev/null
# instead of creating log files. Set by run_backtest_optimized.py when running
# in lightweight WFOV mode (skip_model_save=True).
_SUPPRESS_FILE_LOGS = False


def set_suppress_file_logs(suppress: bool):
    """Enable/disable file log suppression for WFOV lightweight mode."""
    global _SUPPRESS_FILE_LOGS
    _SUPPRESS_FILE_LOGS = suppress


"""
Print helper class
"""


class RedirectStdoutToFile:
    def __init__(self, filename="output.txt", mode="a"):
        self.filename = LOGS_DIR / filename  # Use LOGS_DIR from config
        self.mode = mode
        self.original_stdout = sys.stdout
        self.file = None

    def __enter__(self):
        if _SUPPRESS_FILE_LOGS:
            # WFOV lightweight mode: redirect to /dev/null, create no files
            self.file = open(os.devnull, "w")
        else:
            self.file = open(self.filename, self.mode)
        sys.stdout = self.file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()


"""
Function to calculate annual trading periods based on interval and market type
"""


def calculate_annual_trading_periods(interval: str, market_type: str = "forex") -> int:
    """
    Calculates the estimated annual trading periods based on the given interval for Yahoo Finance data.

    Args:
        interval (str): The interval string (e.g., '1m', '2m', '1h', '1d', '5d', '1wk', '1mo', '3mo').
        market_type (str): 'forex' for 24/5 trading, 'stock' for typical 6.5-hour trading days.
                           This primarily affects intraday calculations.

    Returns:
        int: The estimated number of trading periods in a year.
    """
    # Define yearly base periods based on market type
    if market_type == "forex":
        base_minutes_year = 24 * 60 * 260
        base_days_year = 260
    elif market_type == "stock":
        base_minutes_year = 6.5 * 60 * 252
        base_days_year = 252
    elif market_type == "crypto":
        base_minutes_year = 24 * 60 * 365
        base_days_year = 365
        # Crypto markets are open 24/7, so we use a different base
    else:
        raise ValueError("market_type must be 'forex' or 'stock'")

    base_weeks_year = 52
    base_months_year = 12

    match = re.match(
        r"(\d+)([mhdMwk]+)", interval.lower()
    )  # Convert interval to lower for matching
    if not match:
        raise ValueError(
            f"Invalid interval format: {interval}. Expected format like '1m', '1h', '1d', etc."
        )

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "m":
        return int(base_minutes_year / value)
    elif unit == "h":
        return int(base_minutes_year / (value * 60))
    elif unit == "d":
        return int(base_days_year / value)
    elif unit == "wk":
        return int(base_weeks_year / value)
    elif unit == "mo":
        return int(base_months_year / value)
    else:
        raise ValueError(
            f"Unsupported interval unit: {unit}. Supported: 'm', 'h', 'd', 'wk', 'mo'."
        )
