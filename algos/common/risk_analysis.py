# algos/common/risk_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime, timedelta # Import timedelta
import os
import sys
import re

# Assume project_root_dir is handled by data_loader or a central config
# For standalone script, you'd define it here too.
try:
    current_execution_dir = Path(os.getcwd())
    project_root_dir = current_execution_dir
    for _ in range(5):
        if (project_root_dir / "algos").is_dir():
            break
        if project_root_dir == project_root_dir.parent:
            break
        project_root_dir = project_root_dir.parent
    if not (project_root_dir / "algos").is_dir():
        project_root_dir = current_execution_dir
except NameError:
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root_dir = current_script_dir.parent.parent

logs_dir = project_root_dir / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

# Re-define RedirectStdoutToFile if it's used directly in this module for logging
# (It's already in data_loader, but good practice to ensure consistency or import it)
# Make sure your utils.py (or common/utils.py) exists and contains RedirectStdoutToFile
try:
    from algos.common.utils import RedirectStdoutToFile 
except ImportError:
    # Fallback if the utility cannot be imported (e.g., running this file standalone)
    class RedirectStdoutToFile:
        def __init__(self, filename="output.txt", mode='a'):
            self.filename = logs_dir / filename
            self.mode = mode
            self.original_stdout = sys.stdout
            self.file = None

        def __enter__(self):
            self.file = open(self.filename, self.mode)
            sys.stdout = self.file
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout = self.original_stdout
            if self.file:
                self.file.close()

def calculate_risk_metrics(test_df_results: pd.DataFrame, model_name: str, log_prefix: str) -> dict:
    """
    Calculates and prints various risk metrics for the strategy.

    Args:
        test_df_results (pd.DataFrame): DataFrame containing 'strategy_tc' returns.
        model_name (str): The name of the model.
        log_prefix (str): Prefix for logging output.

    Returns:
        dict: A dictionary of risk metrics.
    """
    if test_df_results.empty or 'strategy_tc' not in test_df_results.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"Error: Insufficient data or missing 'strategy_tc' for risk metrics calculation for {model_name}.")
        return {}

    cleaned_strategy_tc = test_df_results['strategy_tc'].replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned_strategy_tc.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"Warning: 'strategy_tc' is empty after cleaning for risk metrics of {model_name}. Skipping.")
        return {}

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print(f"\n{'=' * 50}")
        print(f"Risk Metrics ({model_name}):")
        print(f"{'=' * 50}")

        metrics = {}

        # Calculate cumulative wealth (equity curve)
        # Assuming cleaned_strategy_tc contains log returns
        # We add a starting point of 1.0 (100% equity) for accurate cumulative calculation
        # The index for this 1.0 should be the day before the first return
        cumulative_wealth = np.exp(cleaned_strategy_tc.cumsum())
        
        # Prepend 1.0 at the effective start date for proper cumulative wealth calculation
        if not cumulative_wealth.empty:
            first_data_date = cumulative_wealth.index[0]
            # Ensure the initial index is a datetime object compatible with the rest
            initial_equity_series = pd.Series([1.0], index=[first_data_date - pd.Timedelta(days=1)])
            cumulative_wealth_adjusted = pd.concat([initial_equity_series, cumulative_wealth])
        else:
            # This case should ideally be caught by cleaned_strategy_tc.empty check
            metrics['max_drawdown'] = np.nan
            metrics['longest_drawdown_period_days'] = np.nan
            print(f"Warning: Cumulative wealth series is empty for {model_name}. Some metrics skipped.")
            return metrics # Return early if cumulative_wealth is empty

        # Calculate previous peaks (high-water mark)
        previous_peaks = cumulative_wealth_adjusted.expanding(min_periods=1).max()

        # Calculate Max Drawdown
        drawdown = (cumulative_wealth_adjusted / previous_peaks) - 1
        max_drawdown = drawdown.min()
        print(f"Max Drawdown: {max_drawdown:.4f}")
        metrics['max_drawdown'] = max_drawdown

        # Calculate Longest Drawdown Period
        longest_drawdown_duration = timedelta(0) # Initialize as timedelta object
        current_drawdown_start_date = None

        # Iterate through the drawdown series to find the longest period from peak to recovery
        for i in range(len(drawdown)):
            current_date = drawdown.index[i]
            
            # If current drawdown is 0, it means we are at a peak or have recovered
            if drawdown.iloc[i] == 0:
                if current_drawdown_start_date is not None:
                    # A drawdown has just ended. Calculate its duration.
                    duration = current_date - current_drawdown_start_date
                    if duration > longest_drawdown_duration:
                        longest_drawdown_duration = duration
                    current_drawdown_start_date = None # Reset, no active drawdown
            else: # drawdown.iloc[i] < 0, meaning we are currently in a drawdown
                if current_drawdown_start_date is None:
                    # This is the start of a new drawdown period.
                    # The start date for this period is the date of the *last peak* encountered.
                    # This is usually the `current_date` if `drawdown.iloc[i-1]` was 0,
                    # or the date of the previous `previous_peaks` value.
                    # Given how `drawdown` series is structured (from peak),
                    # the start date of this drawdown period is the date of the corresponding peak.
                    # We can use the previous_peaks series to find the exact peak date.
                    
                    # Find the date of the actual peak that this drawdown started from.
                    # It's the most recent date `d <= current_date` where `cumulative_wealth_adjusted.loc[d] == previous_peaks.loc[d]`
                    # This is essentially `previous_peaks.index[i]` (the date of the peak up to this point)
                    
                    # For simplicity and correctness with the loop:
                    # The `current_drawdown_start_date` should be the date of the highest point
                    # that the curve was at *before* this specific drop occurred.
                    
                    # If this is the first time we drop, the start date is the previous peak's date.
                    # The last time the `drawdown` was zero, before this current date.
                    
                    # This can be found by looking for the last True in `at_peak` before `i`.
                    
                    # If the drawdown just started (current_drawdown_start_date is None)
                    # The start date is the date of the peak that led to this drawdown.
                    # This means looking back to the last time `drawdown` was 0.
                    last_peak_index_before_current = i - 1
                    while last_peak_index_before_current >= 0 and drawdown.iloc[last_peak_index_before_current] != 0:
                        last_peak_index_before_current -= 1
                    
                    if last_peak_index_before_current >= 0:
                        current_drawdown_start_date = drawdown.index[last_peak_index_before_current]
                    else:
                        # If drawdown starts from the very beginning, use the first date (or the prepended date)
                        current_drawdown_start_date = cumulative_wealth_adjusted.index[0]

        # Handle the case where the drawdown extends to the very end of the data
        if current_drawdown_start_date is not None:
            # If the loop finishes and a drawdown is still active
            # (i.e., `current_drawdown_start_date` is not None after last element processed),
            # calculate duration from `current_drawdown_start_date` to the last date in the series.
            final_duration = drawdown.index[-1] - current_drawdown_start_date
            if final_duration > longest_drawdown_duration:
                longest_drawdown_duration = final_duration

        print(f"Longest Drawdown Period: {longest_drawdown_duration.days} days") # Output in days
        metrics['longest_drawdown_period_days'] = longest_drawdown_duration.days

        # VaR (Value at Risk) - 95% confidence (assuming daily returns)
        # Using historical (empirical) VaR
        if not cleaned_strategy_tc.empty:
            var_95 = cleaned_strategy_tc.quantile(0.05)
        else:
            var_95 = np.nan
        print(f"Daily VaR (95%): {var_95:.4f}")
        metrics['daily_var_95'] = var_95

        # CVaR (Conditional VaR / Expected Shortfall) - 95% confidence
        # Average of returns worse than VaR
        if not cleaned_strategy_tc.empty:
            cvar_95 = cleaned_strategy_tc[cleaned_strategy_tc <= var_95].mean()
        else:
            cvar_95 = np.nan
        print(f"Daily CVaR (95%): {cvar_95:.4f}")
        metrics['daily_cvar_95'] = cvar_95

        # Skewness
        strategy_skewness = cleaned_strategy_tc.skew()
        print(f"Skewness: {strategy_skewness:.4f}")
        metrics['skewness'] = strategy_skewness

        # Kurtosis
        strategy_kurtosis = cleaned_strategy_tc.kurtosis()
        print(f"Kurtosis: {strategy_kurtosis:.4f}")
        metrics['kurtosis'] = strategy_kurtosis

        print(f"{'=' * 50}\n")
        return metrics

def plot_drawdowns(test_df_results: pd.DataFrame, model_name: str, log_prefix: str):
    """
    Generates and saves a plot of the strategy's drawdowns over time.

    Args:
        test_df_results (pd.DataFrame): DataFrame containing 'strategy_tc' returns.
        model_name (str): The name of the model.
        log_prefix (str): Prefix for saving the plot.
    """
    if test_df_results.empty or 'strategy_tc' not in test_df_results.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"Error: Insufficient data or missing 'strategy_tc' for drawdown plot for {model_name}.")
        return

    cleaned_strategy_tc = test_df_results['strategy_tc'].replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned_strategy_tc.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"Warning: 'strategy_tc' is empty after cleaning for drawdown plot of {model_name}. Skipping.")
        return

    # Extract ticker, interval, start, end from log_prefix for plot titles and filenames
    ticker_info = "UNKNOWN"
    interval_info = "UNKNOWN"
    start_info = "UNKNOWN"
    end_info = "UNKNOWN"
    match = re.search(r'([A-Z]+)_(\d+[a-z]+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})', log_prefix)
    if match:
        ticker_info = match.group(1)
        interval_info = match.group(2)
        start_info = match.group(3)
        end_info = match.group(4)

    images_dir = Path(project_root_dir / 'images') # Make sure this points to your images folder
    images_dir.mkdir(parents=True, exist_ok=True)

    # Calculate cumulative wealth (equity curve) for plotting drawdown
    cumulative_wealth = np.exp(cleaned_strategy_tc.cumsum())
    
    # Prepend 1.0 for accurate drawdown plotting from initial equity
    if not cumulative_wealth.empty:
        first_data_date = cumulative_wealth.index[0]
        initial_equity_series = pd.Series([1.0], index=[first_data_date - pd.Timedelta(days=1)])
        cumulative_wealth_adjusted = pd.concat([initial_equity_series, cumulative_wealth])
    else:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(f"Warning: Cumulative wealth series is empty for {model_name}. Drawdown plot skipped.")
        return

    peak = cumulative_wealth_adjusted.expanding(min_periods=1).max()
    drawdown = (cumulative_wealth_adjusted / peak) - 1

    fig, ax = plt.subplots(figsize=(12, 7))
    drawdown.plot(ax=ax, title=f'Strategy Drawdown ({model_name}) - {ticker_info} {interval_info} ({start_info} to {end_info})', color='darkred')
    ax.set_ylabel('Drawdown')
    ax.set_xlabel('Date')
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0)) # Format as percentage

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f'{model_name}_drawdown_plot_{ticker_info}_{interval_info}_{start_info}_{end_info}_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(images_dir / image_filename)
    plt.close(fig)
    
    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print(f"Saved drawdown plot: {images_dir / image_filename}")