# algos/common/config.py

import os
from pathlib import Path
import sys
import random
import torch
import tensorflow as tf


# --- Project Root Directory Detection ---
def get_project_root() -> Path:
    """
    Dynamically determines the project root directory.
    Assumes 'project root' is the root.
    Works for both script execution and Jupyter/IPython environments.
    """
    current_path = Path(os.getcwd())
    # Traverse up to 5 levels to find 'project root'
    for _ in range(5):
        if (current_path / "algos").is_dir():
            return current_path
        if current_path == current_path.parent:  # Reached file system root
            break
        current_path = current_path.parent

    # Fallback if 'project root' is not found
    print(
        "Warning: Could not automatically detect 'project root' project root. "
        "Using current working directory as base.",
        file=sys.stderr,
    )
    return Path(os.getcwd())


PROJECT_ROOT_DIR = get_project_root()

# --- Define Paths ---
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGES_DIR = PROJECT_ROOT_DIR / "images"
LOGS_DIR = PROJECT_ROOT_DIR / "logs"
MODEL_DUMPS_DIR = PROJECT_ROOT_DIR / "algos" / "model_dumps"
PICKLE_DUMPS_DIR = (
    PROJECT_ROOT_DIR / "algos" / "pickle_dumps"
)  # Added based on your tree output

# Create directories if they don't exist
for d in [DATA_DIR, IMAGES_DIR, LOGS_DIR, MODEL_DUMPS_DIR, PICKLE_DUMPS_DIR]:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)

# --- General Configuration ---
RF_RATE = 0.04  # Default risk-free rate (used when no date context available)


def get_risk_free_rate(date=None):
    """Get approximate annualized risk-free rate for a given date.

    Based on historical Federal Funds Rate annual averages.
    For precise work, use FRED:FEDFUNDS time series data.

    Args:
        date: pandas Timestamp, datetime, or None. If None, returns RF_RATE default.

    Returns:
        Annualized risk-free rate as a float.
    """
    if date is None:
        return RF_RATE
    try:
        year = date.year
    except AttributeError:
        return RF_RATE
    # Approximate annual averages from FRED FEDFUNDS effective rate
    rate_by_year = {
        2010: 0.0018,
        2011: 0.0010,
        2012: 0.0014,
        2013: 0.0011,
        2014: 0.0009,
        2015: 0.0013,
        2016: 0.0039,
        2017: 0.0100,
        2018: 0.0191,
        2019: 0.0216,
        2020: 0.0036,
        2021: 0.0008,
        2022: 0.0168,
        2023: 0.0508,
        2024: 0.0483,
        2025: 0.0433,
        2026: 0.0400,
    }
    return rate_by_year.get(year, RF_RATE)


PTC = 0.00035  # Proportional transaction cost

# --- Column Names for Data Loading ---
COL_NAMES = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]

# --- Environment Configuration ---
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.simplefilter("ignore")
plt.style.use("seaborn-v0_8")
# mpl.rcParams['font.family'] = 'serif' # This needs pylab's mpl, moved to utils.py if needed there
np.random.seed(1000)

# Maximum Kelly Leverage Cap for practical application and numerical stability
# This prevents extremely high and often unrealistic Kelly values from dominating plots
# and causing potential calculation issues (e.g., overflow with exp()).
# Adjust this based on your risk tolerance and what constitutes "reasonable" leverage.

# Traditional assets (stocks, ETFs) - conservative leverage
MAX_KELLY_LEVERAGE_CAP = 4.0  # Cap the calculated Kelly at 4x for stability

# Crypto assets - higher leverage available on exchanges (up to 100x on some)
MAX_KELLY_LEVERAGE_CAP_CRYPTO = (
    20.0  # Cap at 20x for crypto (reasonable for most exchanges)
)

# Maximum Kelly Leverage to include in the "reasonable" cumulative return plot.
# Leverages above this will be plotted separately in an "extreme" plot for better visualization.
MAX_KELLY_LEVERAGE_FOR_PLOTTING = 2
MAX_KELLY_LEVERAGE_FOR_PLOTTING_CRYPTO = (
    10  # Higher for crypto due to higher leverage availability
)


def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Crucial for deterministic CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Can slow down training
    # For PyTorch 1.8+
    # torch.use_deterministic_algorithms(True) # This can raise errors if no deterministic impl exists
    tf.random.set_seed(seed)


# Call this at the very beginning of your script/notebook
set_all_seeds(1000)  # Set a fixed seed for reproducibility across all libraries
