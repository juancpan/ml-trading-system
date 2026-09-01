import pandas as pd
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import warnings
import argparse
import logging
from datetime import datetime
import scipy.optimize as sco # Import scipy.optimize

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure plot style for better aesthetics
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (12, 8) # Set a default figure size

# --- Configuration for Directories and Logging ---
# Base directories (relative to where the script is run from)
BASE_SCRIPT_DIR = os.path.dirname(__file__)
BASE_DATA_DIR = os.path.join(BASE_SCRIPT_DIR, '..', '..', 'data')
BASE_IMAGE_DIR = os.path.join(BASE_SCRIPT_DIR, '..', '..', 'images')
BASE_LOG_DIR = os.path.join(BASE_SCRIPT_DIR, '..', '..', 'logs')

# Ensure output directories exist
os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(BASE_IMAGE_DIR, exist_ok=True)
os.makedirs(BASE_LOG_DIR, exist_ok=True)

# Generate a unique timestamp for output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Configure logging to output to both console and a file with a unique timestamp
LOG_FILE_PATH = os.path.join(BASE_LOG_DIR, f'portfolio_optimization_{TIMESTAMP}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(LOG_FILE_PATH) # Output to file
    ]
)

# --- Functions ---

def load_and_preprocess_data(file_path, start_date=None, end_date=None):
    """
    Loads financial data from a CSV, preprocesses it, and calculates logarithmic returns.
    Applies date filtering if start_date and end_date are provided.

    Parameters:
    -----------
    file_path : str
        The path to the CSV file containing financial data.
    start_date : str, optional
        Start date for filtering data (YYYY-MM-DD). If None, no start date filter applied.
    end_date : str, optional
        End date for filtering data (YYYY-MM-DD). If None, no end date filter applied.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the logarithmic returns for each asset. Returns an empty DataFrame
        if the file is not found or other errors occur.
    """
    try:
        # Load the data, parse 'Date' as datetime and set as index
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        logging.info(f"Successfully loaded data from '{file_path}'.")
        logging.info(f"Original data shape: {df.shape}")

        # Drop rows where all values are NaN (e.g., empty rows or header-only rows)
        df = df.dropna(how='all')

        # Apply date filtering if specified
        if start_date:
            df = df.loc[df.index >= start_date]
            logging.info(f"Filtered data from start date {start_date}. New shape: {df.shape}")
        if end_date:
            df = df.loc[df.index <= end_date]
            logging.info(f"Filtered data to end date {end_date}. New shape: {df.shape}")

        # Forward-fill and then backward-fill to handle missing values
        # This assumes missing values should be filled with the last known good value
        df = df.ffill().bfill()

        if df.empty:
            raise ValueError("DataFrame is empty after loading and cleaning or filtering. Check input data and date range.")

        # Calculate logarithmic returns
        log_returns = np.log(df / df.shift(1)).dropna()
        logging.info(f"Logarithmic returns calculated. Shape: {log_returns.shape}")

        if log_returns.empty:
            raise ValueError("Log returns DataFrame is empty after calculation. Check data frequency and completeness.")

        return log_returns

    except FileNotFoundError:
        logging.error(f"Error: The file '{file_path}' was not found. Please ensure the data file exists in '{BASE_DATA_DIR}'.")
        return pd.DataFrame() # Return empty DataFrame to signal failure and abort workflow
    except Exception as e:
        logging.error(f"An error occurred during data loading or preprocessing: {e}", exc_info=True)
        return pd.DataFrame() # Return empty DataFrame to signal failure and abort workflow

def perform_normality_tests(returns_df, output_dir, timestamp):
    """
    Performs normality tests and generates Q-Q plots for each series in the DataFrame.

    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of logarithmic returns, where each column represents an asset.
    output_dir : str
        Directory to save the Q-Q plots.
    timestamp : str
        Unique timestamp for filenames.
    """
    if returns_df.empty:
        logging.warning("No data available to perform normality tests.")
        return

    logging.info(f"\n--- Performing Normality Tests on {len(returns_df.columns)} Assets ---")

    # Determine optimal subplot grid for Q-Q plots
    num_assets = returns_df.shape[1]
    cols = min(4, num_assets)  # Max 4 columns for plots
    rows = (num_assets + cols - 1) // cols # Calculate rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

    for i, column in enumerate(returns_df.columns):
        data = returns_df[column].dropna()
        if data.empty:
            logging.warning(f"Skipping '{column}' due to no valid data for normality test.")
            continue

        logging.info(f"\nResults for '{column}':")
        # Skewness Test
        skew, p_value_skew = scs.skewtest(data)
        logging.info(f"  Skew of data set: {scs.skew(data):>14.4f}")
        logging.info(f"  Skew test p-value: {p_value_skew:>13.4f} ({'Normal' if p_value_skew > 0.05 else 'Not Normal'} at 5% level)")

        # Kurtosis Test
        kurt, p_value_kurt = scs.kurtosistest(data)
        logging.info(f"  Kurtosis of data set: {scs.kurtosis(data):>12.4f}")
        logging.info(f"  Kurtosis test p-value: {p_value_kurt:>9.4f} ({'Normal' if p_value_kurt > 0.05 else 'Not Normal'} at 5% level)")

        # Jarque-Bera Test (Omnibus Test)
        _, p_value_jb = scs.jarque_bera(data)
        logging.info(f"  Jarque-Bera test p-value: {p_value_jb:>7.4f} ({'Normal' if p_value_jb > 0.05 else 'Not Normal'} at 5% level)")

        # Generate Q-Q plot
        ax = axes[i]
        sm.qqplot(data, line='s', ax=ax)
        ax.set_title(f'Q-Q Plot for {column}', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel('Theoretical Quantiles', fontsize=8)
        ax.set_ylabel('Sample Quantiles', fontsize=8)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'qq_plots_all_assets_{timestamp}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close(fig) # Close the figure to free memory
    logging.info(f"\nQ-Q plots saved to '{plot_path}'")


# --- Portfolio Optimization Functions (for scipy.optimize) ---
def statistics(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Calculates portfolio return, volatility, and Sharpe ratio.
    Used as a helper for optimization functions.
    """
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

def min_func_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Objective function to minimize the negative Sharpe ratio.
    """
    return -statistics(weights, mean_returns, cov_matrix, risk_free_rate)[2]

def min_func_volatility(weights, mean_returns, cov_matrix):
    """
    Objective function to minimize the portfolio volatility.
    """
    return statistics(weights, mean_returns, cov_matrix, 0.0)[1] # Risk-free rate is irrelevant for min volatility

# --- Main Portfolio Optimization Function ---
def perform_portfolio_optimization(returns_df, num_portfolios, risk_free_rate, output_dir, timestamp):
    """
    Performs portfolio optimization using scipy.optimize and visualizes the results,
    optionally combining with Monte Carlo simulation for visualization.

    Parameters:
    -----------
    returns_df : pd.DataFrame
        DataFrame of logarithmic returns, where each column represents an asset.
    num_portfolios : int
        Number of random portfolios to simulate for visualization.
    risk_free_rate : float
        Annualized risk-free rate for Sharpe ratio calculation.
    output_dir : str
        Directory to save the portfolio optimization plot.
    timestamp : str
        Unique timestamp for filenames.

    Returns:
    --------
    dict
        A dictionary containing results for max Sharpe and min volatility portfolios.
    """
    if returns_df.empty:
        logging.warning("No data available to perform portfolio optimization.")
        return {}

    logging.info(f"\n--- Performing Portfolio Optimization on {len(returns_df.columns)} Assets ---")

    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(returns_df.columns)

    # --- Define Constraints and Bounds for Optimization ---
    # Constraint: Sum of weights must be 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds: Weights for each asset must be between 0 and 1 (no short-selling)
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Initial guess: Equally weighted portfolio
    initial_weights = np.array(num_assets * [1. / num_assets])

    # --- Optimize for Maximum Sharpe Ratio ---
    logging.info("\n--- Optimizing for Maximum Sharpe Ratio ---")
    optimal_sharpe = sco.minimize(min_func_sharpe, initial_weights,
                                  args=(mean_returns, cov_matrix, risk_free_rate),
                                  method='SLSQP', bounds=bounds, constraints=constraints)
    
    max_sharpe_weights = optimal_sharpe['x']
    max_sharpe_metrics = statistics(max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_return = max_sharpe_metrics[0]
    max_sharpe_volatility = max_sharpe_metrics[1]
    max_sharpe_ratio = max_sharpe_metrics[2]

    logging.info("\nMaximum Sharpe Ratio Portfolio (Optimized):")
    logging.info(f"  Return: {max_sharpe_return:.4f}")
    logging.info(f"  Volatility: {max_sharpe_volatility:.4f}")
    logging.info(f"  Sharpe Ratio: {max_sharpe_ratio:.4f}")
    logging.info("  Weights:")
    for asset, weight in zip(returns_df.columns, max_sharpe_weights):
        logging.info(f"    {asset}: {weight:.4f}")

    # --- Optimize for Minimum Volatility ---
    logging.info("\n--- Optimizing for Minimum Volatility ---")
    optimal_volatility = sco.minimize(min_func_volatility, initial_weights,
                                      args=(mean_returns, cov_matrix),
                                      method='SLSQP', bounds=bounds, constraints=constraints)
    
    min_vol_weights = optimal_volatility['x']
    min_vol_metrics = statistics(min_vol_weights, mean_returns, cov_matrix, risk_free_rate) # Use risk_free_rate here for full metrics
    
    min_vol_return = min_vol_metrics[0]
    min_vol_volatility = min_vol_metrics[1]
    min_vol_sharpe_ratio = min_vol_metrics[2]

    logging.info("\nMinimum Volatility Portfolio (Optimized):")
    logging.info(f"  Return: {min_vol_return:.4f}")
    logging.info(f"  Volatility: {min_vol_volatility:.4f}")
    logging.info(f"  Sharpe Ratio: {min_vol_sharpe_ratio:.4f}")
    logging.info("  Weights:")
    for asset, weight in zip(returns_df.columns, min_vol_weights):
        logging.info(f"    {asset}: {weight:.4f}")

    # --- Monte Carlo Simulation (for visualization of the efficient frontier) ---
    logging.info(f"\n--- Running Monte Carlo Simulation for Visualization ({num_portfolios} portfolios) ---")
    results = np.zeros((3, num_portfolios)) # For return, volatility, sharpe ratio
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) # Normalize weights

        port_metrics = statistics(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0,i] = port_metrics[1] # Volatility
        results[1,i] = port_metrics[0] # Return
        results[2,i] = port_metrics[2] # Sharpe Ratio

    portfolio_results_df = pd.DataFrame(results.T, columns=['Volatility', 'Return', 'Sharpe_Ratio'])

    # --- Plotting the efficient frontier with optimized portfolios ---
    plt.figure(figsize=(12, 8))
    plt.scatter(portfolio_results_df['Volatility'], portfolio_results_df['Return'],
                c=portfolio_results_df['Sharpe_Ratio'], cmap='viridis', marker='o', s=10, alpha=0.6,
                label='Random Portfolios')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Portfolio Optimization: Efficient Frontier')
    plt.xlabel('Annualized Volatility (Standard Deviation)')
    plt.ylabel('Annualized Return')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Highlight Max Sharpe Ratio portfolio from scipy.optimize
    plt.scatter(max_sharpe_volatility, max_sharpe_return,
                marker='*', color='red', s=500, label='Maximum Sharpe Ratio Portfolio (Optimized)')

    # Highlight Minimum Volatility portfolio from scipy.optimize
    plt.scatter(min_vol_volatility, min_vol_return,
                marker='*', color='blue', s=500, label='Minimum Volatility Portfolio (Optimized)')

    plt.legend(labelspacing=0.8)
    plot_path = os.path.join(output_dir, f'portfolio_optimization_{timestamp}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close() # Close the figure to free memory
    logging.info(f"\nPortfolio optimization plot saved to '{plot_path}'")

    return {
        'max_sharpe_optimized': {
            'Return': max_sharpe_return,
            'Volatility': max_sharpe_volatility,
            'Sharpe Ratio': max_sharpe_ratio,
            'Weights': max_sharpe_weights
        },
        'min_volatility_optimized': {
            'Return': min_vol_return,
            'Volatility': min_vol_volatility,
            'Sharpe Ratio': min_vol_sharpe_ratio,
            'Weights': min_vol_weights
        }
    }

# --- Main Workflow Execution ---

def main_workflow(args):
    """
    Main function to run the complete financial analysis workflow.
    Accepts arguments for start_date, end_date, and interval.
    """
    logging.info("Starting financial analysis workflow...")

    # Construct DATA_FILE_PATH using parsed arguments for dynamic filename
    DATA_FILE_PATH = os.path.join(BASE_DATA_DIR, f'financial_data_combined_prices_{args.start}_{args.end}_{args.interval}.csv')
    logging.info(f"Attempting to load data from: {DATA_FILE_PATH}")

    # Step 1: Load and Preprocess Data
    log_returns = load_and_preprocess_data(DATA_FILE_PATH, args.start, args.end)
    if log_returns.empty:
        logging.error("Workflow terminated due to data loading/preprocessing failure.")
        return

    # Step 2: Perform Normality Tests
    perform_normality_tests(log_returns, output_dir=BASE_IMAGE_DIR, timestamp=TIMESTAMP)

    # Step 3: Perform Portfolio Optimization using scipy.optimize.minimize
    perform_portfolio_optimization(log_returns, num_portfolios=25000, risk_free_rate=0.01, output_dir=BASE_IMAGE_DIR, timestamp=TIMESTAMP)
    
    logging.info("\nFinancial analysis workflow completed.")
    logging.info(f"All logs saved to: {LOG_FILE_PATH}")
    logging.info(f"All plots saved to: {BASE_IMAGE_DIR} with timestamp '{TIMESTAMP}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform normality tests and portfolio optimization on financial data.")
    parser.add_argument('--start', type=str, default='2019-01-01',
                        help="Start date for data analysis (YYYY-MM-DD). Default: 2019-01-01")
    parser.add_argument('--end', type=str, default='2025-06-01',
                        help="End date for data analysis (YYYY-MM-DD). Default: 2025-06-01")
    parser.add_argument('--interval', type=str, default='1d',
                        help="Data interval (e.g., '1d' for daily). Note: This script expects daily data in CSV matching the filename pattern. Default: 1d")

    args = parser.parse_args()

    logging.info(f"Script called with: Start Date={args.start}, End Date={args.end}, Interval={args.interval}")
    
    main_workflow(args)