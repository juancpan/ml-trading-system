import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Define parameters from the user's log
START_DATE = '2017-01-01'
END_DATE = '2025-06-01'
INTERVAL = '1d' # Daily interval

# MC Portfolio #2 weights from the user's log
MC_PORTFOLIO_2_WEIGHTS = {
    'SPY': 0.0447,
    'GDX': 0.0763,
    'GLD': 0.1526,
    'NVDA': 0.1640,
    'TSLA': 0.0465,
    'AAPL': 0.0761,
    'AMZN': 0.0207,
    'QQQ': 0.0404,
    'CRM': 0.0113,
    'EURUSD': 0.0099,
    'GBPUSD': 0.0135,
    'GOOGL': 0.0893,
    'IVV': 0.0416,
    'MA': 0.0975,
    'META': 0.0052,
    'USDJPY': 0.1103
}

# Define user-defined equity balance
DEFAULT_EQUITY_BALANCE = 10000.0

# Generate a unique timestamp for output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILENAME = f"portfolio_ohlc_{TIMESTAMP}.csv"

# Directory to save the output CSV (using the base data directory from previous scripts)
BASE_SCRIPT_DIR = os.path.dirname(__file__)
BASE_DATA_DIR = os.path.join(BASE_SCRIPT_DIR, '..', '..', 'data')
os.makedirs(BASE_DATA_DIR, exist_ok=True)

print(f"Starting portfolio OHLC calculation...")

# --- Step 1: Fetch all data and prepare for common date alignment ---
all_ticker_raw_data = {}
all_indices = []

for ticker in MC_PORTFOLIO_2_WEIGHTS.keys():
    # Adjust ticker for currency pairs for yfinance
    yf_ticker = ticker
    if ticker in ['EURUSD', 'GBPUSD', 'USDJPY']:
        yf_ticker = f"{ticker}=X"
    # elif ticker == 'USDJPY':
    #     yf_ticker = "JPY=X" # USDJPY is represented by JPY=X for its inverse movement
    
    try:
        data = yf.download(yf_ticker, start=START_DATE, end=END_DATE, interval=INTERVAL, auto_adjust=False)
        if not data.empty:
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            # Filter to the exact date range specified by the user
            data_filtered = data[(data.index >= pd.to_datetime(START_DATE)) & (data.index < pd.to_datetime(END_DATE))]

            if not data_filtered.empty:
                all_ticker_raw_data[ticker] = data_filtered
                all_indices.append(data_filtered.index)
                print(f"Successfully fetched and filtered data for {ticker} ({yf_ticker}). Shape: {data_filtered.shape}")
            else:
                print(f"No data for {ticker} ({yf_ticker}) within the specified date range. Skipping.")
        else:
            print(f"No data fetched for {ticker} ({yf_ticker}). Skipping.")
    except Exception as e:
        print(f"Error fetching data for {ticker} ({yf_ticker}): {e}")

if not all_ticker_raw_data:
    raise ValueError("No valid ticker data was fetched. Cannot create portfolio.")

# --- Step 2: Determine the intersection of all trading dates ---
if not all_indices:
    raise ValueError("No valid date indices found from fetched data.")

# Start with the first index and intersect with the rest
final_common_dates = all_indices[0]
for i in range(1, len(all_indices)):
    final_common_dates = final_common_dates.intersection(all_indices[i])

if final_common_dates.empty:
    raise ValueError("No common trading dates found across all assets within the specified range. Adjust date range or assets.")

print(f"\nCalculated {len(final_common_dates)} common trading dates from {final_common_dates.min().strftime('%Y-%m-%d')} to {final_common_dates.max().strftime('%Y-%m-%d')}")

# --- Step 3: Initialize portfolio series with common dates ---
portfolio_open = pd.Series(0.0, index=final_common_dates)
portfolio_high = pd.Series(0.0, index=final_common_dates)
portfolio_low = pd.Series(0.0, index=final_common_dates)
portfolio_close = pd.Series(0.0, index=final_common_dates)
portfolio_adj_close = pd.Series(0.0, index=final_common_dates)

# --- Step 4: Process each ticker and aggregate to portfolio ---
for ticker, weight in MC_PORTFOLIO_2_WEIGHTS.items():
    if ticker not in all_ticker_raw_data:
        print(f"Skipping {ticker} as no raw data was fetched or filtered for it.")
        continue

    df_raw = all_ticker_raw_data[ticker]
    
    # Determine the effective price column for normalization
    effective_price_col = 'Adj Close' if 'Adj Close' in df_raw.columns else 'Close'

    if effective_price_col not in df_raw.columns:
        print(f"Skipping {ticker} due to missing {effective_price_col} column in raw data.")
        continue

    # Define columns to extract and reindex
    cols_to_extract = ['Open', 'High', 'Low', 'Close']
    if 'Adj Close' in df_raw.columns:
        cols_to_extract.append('Adj Close')

    # --- REINVENTED DATA ALIGNMENT & NUMERIC CONVERSION START ---
    # Create an empty DataFrame with the common dates as index and target columns
    df_aligned = pd.DataFrame(index=final_common_dates, columns=cols_to_extract)

    # Populate df_aligned column by column with robust conversion
    for col in cols_to_extract:
        if col in df_raw.columns:
            # Reindex the raw Series for this column to common dates
            temp_series = df_raw[col].reindex(final_common_dates)
            
            # --- FIX START: Manual element-wise conversion for extreme robustness ---
            converted_values = []
            if not temp_series.empty:
                for val in temp_series.values:
                    try:
                        # Attempt to convert each value to float
                        converted_values.append(float(val))
                    except (ValueError, TypeError):
                        # If conversion fails (e.g., non-numeric string), append NaN
                        converted_values.append(np.nan)
            
            # Assign the list of converted values (now all floats or NaNs) to the column
            # This explicitly creates a NumPy array from the list.
            df_aligned[col] = np.array(converted_values, dtype=float)
            # --- FIX END ---
        else:
            # If a column is missing in raw data, fill with NaN
            print(f"Warning: Column '{col}' not found in raw data for ticker '{ticker}'. Column will be all NaNs for this ticker.")
            df_aligned[col] = np.nan 

    # Now, df_aligned has numeric values or NaNs. Apply ffill/bfill/fillna(0)
    df_aligned = df_aligned.ffill().bfill().fillna(0)
    # --- REINVENTED DATA ALIGNMENT & NUMERIC CONVERSION END ---

    # After filling NaNs with 0, check if the effective price column is entirely zero or NaN
    if df_aligned[effective_price_col].isnull().all() or (df_aligned[effective_price_col] == 0).all():
        print(f"Warning: All values in '{effective_price_col}' for '{ticker}' are NaN or zero after alignment and filling. Skipping this ticker.")
        continue

    # Get the first non-zero price for normalization
    first_day_price = df_aligned.iloc[0][effective_price_col]
    
    # Ensure it's a scalar
    if isinstance(first_day_price, pd.Series): 
        if first_day_price.empty:
            print(f"Warning: First price for '{ticker}' is an empty Series. Skipping '{ticker}'.")
            continue
        first_day_price = first_day_price.item()

    # Final check for NaN or zero after extraction
    if pd.isna(first_day_price) or first_day_price == 0:
        print(f"Warning: First '{effective_price_col}' value for '{ticker}' is NaN or zero after processing. Skipping normalization and weighted average for this asset.")
        continue

    # Normalize data relative to the first day's price
    normalized_data = df_aligned / first_day_price

    # Aggregate weighted normalized data to portfolio series
    if 'Open' in normalized_data.columns:
        portfolio_open += normalized_data['Open'] * weight
    if 'High' in normalized_data.columns:
        portfolio_high += normalized_data['High'] * weight
    if 'Low' in normalized_data.columns:
        portfolio_low += normalized_data['Low'] * weight
    if 'Close' in normalized_data.columns:
        portfolio_close += normalized_data['Close'] * weight
    
    # Portfolio Adj Close computation
    if 'Adj Close' in normalized_data.columns:
        portfolio_adj_close += normalized_data['Adj Close'] * weight
    elif 'Close' in normalized_data.columns: 
        portfolio_adj_close += normalized_data['Close'] * weight

# Scale the normalized portfolio prices by the initial equity balance
initial_portfolio_value = DEFAULT_EQUITY_BALANCE

# Create the final portfolio DataFrame
portfolio_df = pd.DataFrame({
    'Open': portfolio_open * initial_portfolio_value,
    'High': portfolio_high * initial_portfolio_value,
    'Low': portfolio_low * initial_portfolio_value,
    'Close': portfolio_close * initial_portfolio_value,
    'Adj Close': portfolio_adj_close * initial_portfolio_value
}, index=final_common_dates) 

# Reorder columns as requested: Adj Close, Close, High, Low, Open
portfolio_df = portfolio_df[['Adj Close', 'Close', 'High', 'Low', 'Open']]

# Set the index name to 'Date'
portfolio_df.index.name = 'Date'

# Prepare to write to CSV with custom header rows
output_path = os.path.join(BASE_DATA_DIR, OUTPUT_FILENAME)

with open(output_path, 'w') as f:
    # First row: Price categories
    f.write("Price,Adj Close,Close,High,Low,Open\n")
    
    # Second row: Ticker names
    ticker_name = f"portfolio_{TIMESTAMP}"
    f.write(f"Ticker,{ticker_name},{ticker_name},{ticker_name},{ticker_name},{ticker_name}\n")
    
    # Write the DataFrame, excluding the header and index for now, and appending to the file
    portfolio_df.to_csv(f, header=True, index=True)

print(f"\nFinancial analysis workflow completed. All logs saved to: [logs directory, if applicable]")
print(f"Portfolio OHLC data saved to {output_path}")

# Display the first few rows for confirmation
print("\nFirst 10 rows of the generated portfolio OHLC data:")
with open(output_path, 'r') as f:
    for i, line in enumerate(f):
        if i < 12: 
            print(line.strip())
        else:
            break