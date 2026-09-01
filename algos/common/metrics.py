import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # For currency formatting
from scipy.stats import gmean
import re  # For extracting info from log_prefix
from datetime import datetime
from pathlib import Path
import sys

# Import common utilities and configuration
from algos.common import config
from algos.common.utils import RedirectStdoutToFile

# Set seeds for reproducibility (though plotting itself isn't stochastic,
# this is good practice for any numerical operations)
np.random.seed(1000)
# No tf.random.set_seed here as this file generally doesn't directly use TensorFlow


def calculate_strategy_performance(
    test_df_results: pd.DataFrame,
    model_name: str,
    log_prefix: str,
    max_leverage: float = None,
    no_plots: bool = False,
    bh_returns: pd.Series = None,
) -> dict:
    """
    Calculates and prints performance metrics for a given strategy.
    Also generates and saves cumulative return plots (unless no_plots=True).

    Args:
        test_df_results (pd.DataFrame): DataFrame containing 'returns', 'strategy', 'strategy_tc',
                                         and 'position' columns from the backtest.
                                         Expected to have 'annual_trading_periods' in its .attrs.
        model_name (str): The name of the model (e.g., 'DQN', 'OLS').
        log_prefix (str): The prefix used for logging, contains info like ticker, interval, dates.
        max_leverage (float, optional): Maximum leverage cap for Kelly criterion. If None, uses defaults from config.
        no_plots (bool): If True, skip generating plots/images. Default: False.

    Returns:
        dict: A dictionary of performance metrics.
    """
    if test_df_results.empty or "strategy_tc" not in test_df_results.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Error: Insufficient data or missing 'strategy_tc' column for performance calculation for {model_name}. Skipping metrics."
            )
        return {}

    # Extract ticker, interval, start, end from log_prefix for plot titles and filenames
    ticker_info = "UNKNOWN"
    interval_info = "UNKNOWN"
    start_info = "UNKNOWN"
    end_info = "UNKNOWN"

    # Updated pattern to handle tickers like BTC-USD, ETH-USD
    match = re.search(
        r"([A-Z]+(?:-[A-Z]+)?)_(\d+[a-z]+)_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})",
        log_prefix,
    )
    if match:
        ticker_info = match.group(1)
        interval_info = match.group(2)
        start_info = match.group(3)
        end_info = match.group(4)

    annual_trading_periods = test_df_results.attrs.get(
        "annual_trading_periods", 252
    )  # Default for daily data
    risk_free_rate = config.RF_RATE
    ptc = config.PTC

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print(f"\n{'=' * 50}")
        print(f"Strategy Performance Metrics ({model_name}):")
        print(f"{'=' * 50}")

        # Ensure no NaNs or Infs in strategy_tc before calculations
        # This is critical for log returns and statistical measures
        cleaned_strategy_tc = (
            test_df_results["strategy_tc"].replace([np.inf, -np.inf], np.nan).dropna()
        )
        if cleaned_strategy_tc.empty:
            print(
                f"Warning: 'strategy_tc' is empty after cleaning for {model_name}. Cannot calculate full metrics."
            )
            return {}

        # 1. Annualized Strategy Return
        # For log returns, sum and then annualize
        total_log_return = cleaned_strategy_tc.sum()
        num_periods = len(cleaned_strategy_tc)

        # Avoid division by zero if num_periods is 0 or 1
        if num_periods > 0:
            annualized_strategy_return = (
                np.exp(total_log_return / num_periods * annual_trading_periods) - 1
            )
        else:
            annualized_strategy_return = np.nan

        print(f"Annualized Strategy Return: {annualized_strategy_return:.4f}")

        # 2. Annualized Strategy Volatility
        annualized_strategy_volatility = cleaned_strategy_tc.std() * np.sqrt(
            annual_trading_periods
        )
        print(f"Annualized Strategy Volatility: {annualized_strategy_volatility:.4f}")

        # 3. Strategy Sharpe Ratio
        # Convert annualized risk-free rate to per-period log rate (needed for
        # Sharpe, Kelly, and Buy-and-Hold benchmark calculations below).
        log_rf_rate_per_period = np.log(1 + risk_free_rate) / annual_trading_periods

        if annualized_strategy_volatility > 1e-9:  # Avoid division by zero
            # For log returns, excess return is (mean_log_return_per_period - log_rf_rate_per_period)
            excess_return_per_period = (
                cleaned_strategy_tc.mean() - log_rf_rate_per_period
            )
            strategy_sharpe_ratio = (
                excess_return_per_period
                * np.sqrt(annual_trading_periods)
                / cleaned_strategy_tc.std()
            )
        else:
            strategy_sharpe_ratio = np.nan
        print(f"Strategy Sharpe Ratio: {strategy_sharpe_ratio:.4f}")

        # 4. Kelly Criterion (Full and Half)
        strategy_mean_return = cleaned_strategy_tc.mean()
        strategy_variance = cleaned_strategy_tc.var()

        # Kelly Formula: f = (E[R] - r) / Var[R]
        # Using continuous (log) returns:
        # If strategy_mean_return represents mean log return per period, and strategy_variance is variance of log returns per period
        # then (strategy_mean_return - log_rf_rate_per_period) / strategy_variance is a common form of Kelly for log returns.

        # Ensure positive variance for Kelly calculation
        if strategy_variance > 1e-9:
            # Using the simplified Kelly for log returns based on excess mean and variance
            kelly_f_val_uncapped = (
                strategy_mean_return - log_rf_rate_per_period
            ) / strategy_variance
        else:
            kelly_f_val_uncapped = np.nan

        # Cap the Kelly leverage for practicality in plotting/reporting
        # Use different caps for crypto vs traditional assets
        # Check if ticker is crypto (contains BTC, ETH, SOL, DOGE, or ends with -USD for crypto pairs)
        is_crypto = (
            any(
                crypto in ticker_info.upper()
                for crypto in ["BTC", "ETH", "SOL", "DOGE", "COIN"]
            )
            or ticker_info.endswith("-USD")
            or ticker_info.endswith("USDT")
        )

        # Use provided max_leverage if available, otherwise use defaults from config
        if max_leverage is not None:
            leverage_cap = max_leverage
            print(f"Using custom leverage cap: {leverage_cap}x")
        elif is_crypto:
            leverage_cap = config.MAX_KELLY_LEVERAGE_CAP_CRYPTO
            print(f"Using crypto leverage cap: {leverage_cap}x")
        else:
            leverage_cap = config.MAX_KELLY_LEVERAGE_CAP
            print(f"Using default leverage cap: {leverage_cap}x")

        kelly_f_val = np.clip(kelly_f_val_uncapped, -leverage_cap, leverage_cap)

        strategy_k_lev_full = kelly_f_val
        strategy_k_lev_half = strategy_k_lev_full / 2

        print(f"Optimal Strategy Kelly Leverage (Full): {strategy_k_lev_full:.4f}")
        print(f"Half Strategy Kelly Leverage: {strategy_k_lev_half:.4f}")

        # 5. Buy-and-Hold Benchmark Metrics
        # Use canonical B&H returns if provided (model-independent test window),
        # otherwise fall back to model's test_df_results["returns"]
        raw_bh = bh_returns if bh_returns is not None else test_df_results["returns"]
        cleaned_returns = raw_bh.replace([np.inf, -np.inf], np.nan).dropna()
        if len(cleaned_returns) > 0:
            bh_total_log_return = cleaned_returns.sum()
            bh_annualized_return = (
                np.exp(
                    bh_total_log_return / len(cleaned_returns) * annual_trading_periods
                )
                - 1
            )
            bh_annualized_volatility = cleaned_returns.std() * np.sqrt(
                annual_trading_periods
            )
            if bh_annualized_volatility > 1e-9:
                bh_excess_per_period = cleaned_returns.mean() - log_rf_rate_per_period
                bh_sharpe_ratio = (
                    bh_excess_per_period
                    * np.sqrt(annual_trading_periods)
                    / cleaned_returns.std()
                )
            else:
                bh_sharpe_ratio = np.nan

            # Excess metrics: strategy vs buy-and-hold
            excess_return = annualized_strategy_return - bh_annualized_return
            excess_sharpe = (
                strategy_sharpe_ratio - bh_sharpe_ratio
                if not np.isnan(bh_sharpe_ratio)
                else np.nan
            )

            # Information Ratio: excess return / tracking error
            tracking_diff = (
                cleaned_strategy_tc.reindex(cleaned_returns.index) - cleaned_returns
            )
            tracking_diff = tracking_diff.dropna()
            tracking_error = (
                tracking_diff.std() * np.sqrt(annual_trading_periods)
                if len(tracking_diff) > 1
                else np.nan
            )
            information_ratio = (
                excess_return / tracking_error
                if tracking_error and tracking_error > 1e-9
                else np.nan
            )
        else:
            bh_annualized_return = np.nan
            bh_annualized_volatility = np.nan
            bh_sharpe_ratio = np.nan
            excess_return = np.nan
            excess_sharpe = np.nan
            information_ratio = np.nan

        print(f"\n{'=' * 50}")
        print(f"Buy-and-Hold Benchmark ({model_name}):")
        print(f"{'=' * 50}")
        print(f"B&H Annualized Return: {bh_annualized_return:.4f}")
        print(f"B&H Annualized Volatility: {bh_annualized_volatility:.4f}")
        print(f"B&H Sharpe Ratio: {bh_sharpe_ratio:.4f}")
        print(f"Excess Return (Strategy - B&H): {excess_return:+.4f}")
        print(f"Excess Sharpe (Strategy - B&H): {excess_sharpe:+.4f}")
        print(f"Information Ratio: {information_ratio:.4f}")
        beats_bh = (
            "YES"
            if excess_sharpe and not np.isnan(excess_sharpe) and excess_sharpe > 0
            else "NO"
        )
        print(f"ML Beats Buy-and-Hold: {beats_bh}")
        print(f"{'=' * 50}\n")

        # Store metrics in a dictionary
        metrics = {
            "annualized_strategy_return": annualized_strategy_return,
            "annualized_strategy_volatility": annualized_strategy_volatility,
            "strategy_sharpe_ratio": strategy_sharpe_ratio,
            "optimal_strategy_kelly_leverage_full": strategy_k_lev_full,
            "half_strategy_kelly_leverage": strategy_k_lev_half,
            "bh_annualized_return": bh_annualized_return,
            "bh_annualized_volatility": bh_annualized_volatility,
            "bh_sharpe_ratio": bh_sharpe_ratio,
            "excess_return": excess_return,
            "excess_sharpe": excess_sharpe,
            "information_ratio": information_ratio,
        }

        print(f"{'=' * 50}\n")

        # --- Generate Cumulative Return Plots (skip if no_plots=True) ---
        if not no_plots:
            print("Generating cumulative return plots...")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            images_dir = Path("images")
            images_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure images directory exists

            # Calculate leveraged strategies for plotting
            plot_df = pd.DataFrame(index=test_df_results.index)
            plot_df["Market"] = test_df_results["returns"].cumsum().apply(np.exp)
            plot_df["Strategy (No TC)"] = (
                test_df_results["strategy"].cumsum().apply(np.exp)
            )
            plot_df["Strategy (with TC)"] = (
                test_df_results["strategy_tc"].cumsum().apply(np.exp)
            )

            # Add Kelly leveraged strategies, ensuring they are finite and within reasonable bounds for the first plot
            kelly_leverages_to_plot = []
            if np.isfinite(strategy_k_lev_full):
                kelly_leverages_to_plot.append(strategy_k_lev_full)
            if np.isfinite(strategy_k_lev_half):
                kelly_leverages_to_plot.append(strategy_k_lev_half)
            # Add a 2/3 Kelly
            if np.isfinite(strategy_k_lev_full * 2 / 3):
                kelly_leverages_to_plot.append(strategy_k_lev_full * 2 / 3)

            # Sort to ensure consistent plotting order
            kelly_leverages_to_plot = sorted(
                list(set(kelly_leverages_to_plot)), reverse=True
            )  # Remove duplicates and sort

            reasonable_leverages_for_plot = []
            extreme_leverages_for_plot = []

            # Use appropriate plotting threshold based on asset type
            plotting_threshold = (
                config.MAX_KELLY_LEVERAGE_FOR_PLOTTING_CRYPTO
                if is_crypto
                else config.MAX_KELLY_LEVERAGE_FOR_PLOTTING
            )

            for lev in kelly_leverages_to_plot:
                if lev <= plotting_threshold:
                    reasonable_leverages_for_plot.append(lev)
                else:
                    extreme_leverages_for_plot.append(lev)

            # Generate columns for all potential Kelly leverages
            for lev in kelly_leverages_to_plot:
                col_name = f"Strategy (L={lev:.2f}) (with TC)"
                # Ensure no division by zero or large number issues
                if (
                    test_df_results["strategy_tc"].min() > -1
                ):  # Avoid issues with exp(very large negative)
                    plot_df[col_name] = (
                        (test_df_results["strategy_tc"] * lev).cumsum().apply(np.exp)
                    )
                else:
                    plot_df[col_name] = (
                        (test_df_results["strategy_tc"] * lev)
                        .cumsum()
                        .apply(np.exp)
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(method="ffill")
                    )  # Handle potential -inf issue

            # --- Plot 1: Market, Strategy (No TC), Strategy (with TC) and Reasonable Kelly Leverages ---
            fig1, ax1 = plt.subplots(figsize=(12, 7))

            plot_df[["Market", "Strategy (No TC)", "Strategy (with TC)"]].plot(ax=ax1)
            legend_labels_ax1 = ["Market", "Strategy (No TC)", "Strategy (with TC)"]

            for lev in reasonable_leverages_for_plot:
                col_name = f"Strategy (L={lev:.2f}) (with TC)"
                if col_name in plot_df.columns:
                    plot_df[col_name].plot(ax=ax1)
                    legend_labels_ax1.append(col_name)

            ax1.set_title(
                f"Cumulative Returns ({model_name} Strategy) - {ticker_info} {interval_info} ({start_info} to {end_info})"
            )
            ax1.set_ylabel("Cumulative Returns")
            ax1.set_xlabel("Date")
            ax1.legend(legend_labels_ax1)
            ax1.grid(True)
            # Format y-axis as percentage/currency
            formatter = mticker.FormatStrFormatter("$%.2f")
            ax1.yaxis.set_major_formatter(formatter)

            image_filename_base_strategy = f"{model_name}_cumulative_returns_base_{ticker_info}_{interval_info}_{start_info}_{end_info}_{timestamp}.png"
            plt.tight_layout()  # Adjust layout to prevent labels overlapping
            plt.savefig(images_dir / image_filename_base_strategy)
            plt.close(fig1)
            print(f"Saved plot: {images_dir / image_filename_base_strategy}")

            # --- Plot 2: Extreme Kelly Leverages (if any) ---
            if extreme_leverages_for_plot:
                fig2, ax2 = plt.subplots(figsize=(12, 7))

                # Plot base market return for reference, but it will look flat
                plot_df["Market"].plot(ax=ax2, color="gray", linestyle="--")
                legend_labels_ax2 = ["Market (Reference)"]

                for lev in extreme_leverages_for_plot:
                    col_name = f"Strategy (L={lev:.2f}) (with TC)"
                    if col_name in plot_df.columns:
                        plot_df[col_name].plot(ax=ax2)
                        legend_labels_ax2.append(col_name)

                ax2.set_title(
                    f"Cumulative Returns ({model_name} Strategy) - Extreme Kelly Leverages ({ticker_info} {interval_info})"
                )
                ax2.set_ylabel("Cumulative Returns")
                ax2.set_xlabel("Date")
                ax2.legend(legend_labels_ax2)
                ax2.grid(True)
                ax2.yaxis.set_major_formatter(formatter)  # Apply currency formatter

                image_filename_extreme_strategy = f"{model_name}_cumulative_returns_extreme_kelly_{ticker_info}_{interval_info}_{start_info}_{end_info}_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(images_dir / image_filename_extreme_strategy)
                plt.close(fig2)
                print(f"Saved plot: {images_dir / image_filename_extreme_strategy}")
            else:
                print("No extreme Kelly leverages to plot separately.")
        else:
            print("Skipping plot generation (--no-plots)")

        return metrics
