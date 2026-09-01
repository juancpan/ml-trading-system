import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle  # For saving/loading the statsmodels SARIMAXResultsWrapper object
import sys  # For sys.exit

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage


def run_sarimax_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    lags: int = 5,
    order: tuple = (1, 0, 1),
    seasonal_order: tuple = (0, 0, 0, 0),
    log_prefix: str = "SARIMAX_model",
    **kwargs,
):
    """
    Implements a SARIMAX-based trading strategy using a walk-forward validation approach.
    The model predicts future returns based on historical returns and lagged returns as exogenous features,
    then converts these predictions into trading signals.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
        initial_train_split_ratio (float): Ratio for the initial training data size before walk-forward.
        lags (int): Number of previous time steps (lagged returns) to use as exogenous features.
        order (tuple): Non-seasonal (p, d, q) order of the SARIMAX model.
        seasonal_order (tuple): Seasonal (P, D, Q, S) order of the SARIMAX model.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'position' (trading signals),
                                         'strategy', 'strategy_tc'.
               final_model_object: The last fitted SARIMAXResultsWrapper object.
    """
    # --- Input Data Validation ---
    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting."
            )
        sys.exit(1)

    if "returns" not in data.columns or "direction" not in data.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data for {log_prefix} is missing 'returns' or 'direction' columns. Exiting."
            )
        sys.exit(1)

    # --- Feature Engineering: Create Lagged Returns (Exogenous Features) ---
    # SARIMAX can use exogenous variables, so we'll pass the lagged returns as `exog`.
    cols = []
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
        data[col] = data["returns"].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)  # Drop rows with NaN values introduced by shifting

    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: 'data' DataFrame is empty after creating lags and dropping NaNs for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Train/Test Split for Walk-Forward Validation ---
    # The 'train' set becomes the initial history, 'test' set is for walk-forward predictions.
    split = int(len(data) * initial_train_split_ratio)
    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    if train.empty or test.empty or len(cols) == 0:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Insufficient data for {log_prefix} training or testing. Check data loading and preprocessing steps."
            )
        sys.exit(1)

    # --- Scaling Exogenous Features ---
    scaler = StandardScaler()
    # Fit scaler on training exogenous features, then transform both train and test
    train_exog_scaled = scaler.fit_transform(train[cols])
    test_exog_scaled = scaler.transform(test[cols])

    # Convert scaled features back to DataFrame for easier indexing with history
    train_exog_df = pd.DataFrame(train_exog_scaled, index=train.index, columns=cols)
    test_exog_df = pd.DataFrame(test_exog_scaled, index=test.index, columns=cols)

    # --- Walk-Forward Validation for SARIMAX ---
    predictions_list = []
    model_fit = None  # To store the last successfully fitted model for returning

    # Initialize history with training data
    history_endog = train["returns"].tolist()
    history_exog = train_exog_df.values.tolist()

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Training and Forecasting (Walk-Forward):\n")

        # Loop through the test set, fitting and forecasting one step at a time
        for i in range(len(test)):
            try:
                # Get the current endogenous and exogenous history
                current_endog = history_endog
                current_exog = (
                    np.array(history_exog) if history_exog else None
                )  # Ensure it's numpy array or None

                # Define the SARIMAX model for the current history
                model = SARIMAX(
                    current_endog,
                    exog=current_exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                # Fit the model
                # disp=False suppresses verbose output during fitting
                # low_memory=True can help with memory usage for large datasets
                model_fit = model.fit(disp=False, low_memory=True)

                # Prepare exogenous data for the forecast step
                # SARIMAX forecast expects a 2D array for exog, even for a single future step
                future_exog_for_forecast = test_exog_df.iloc[[i]].values

                # Make one-step-ahead forecast
                forecast_output = model_fit.forecast(
                    steps=1, exog=future_exog_for_forecast
                )
                predicted_return = forecast_output.iloc[
                    0
                ]  # Extract the scalar forecast value
                predictions_list.append(predicted_return)

            except Exception as e:
                # Log the error but continue by appending NaN, allowing the loop to proceed
                print(
                    f"Warning: {log_prefix} model fit or forecast failed at step {i} with error: {e}. Appending NaN."
                )
                predictions_list.append(np.nan)
                model_fit = None  # Reset model_fit if an error occurred

            # Append the actual observation from the test set to the history for the next iteration
            history_endog.append(test["returns"].iloc[i])
            history_exog.append(
                test_exog_df.iloc[i].values.tolist()
            )  # Add as list for list of lists

        # Add predictions to the test DataFrame
        test["predicted_returns"] = pd.Series(predictions_list, index=test.index)
        # Drop rows where prediction might have failed (NaNs introduced)
        test.dropna(subset=["predicted_returns"], inplace=True)

        print("=" * 50 + "\n")

    # --- Convert Predicted Returns to Trading Signals (Position) ---
    # If predicted_returns > 0, then position is 1 (go long), else -1 (go short).
    test["position"] = np.where(test["predicted_returns"] > 0, 1, -1)

    # Ensure the 'direction' column is correctly defined for accuracy_score
    test["direction"] = np.where(test["returns"] > 0, 1, -1)
    # Drop any rows where 'direction' or 'position' might be missing after previous operations
    test.dropna(subset=["direction", "position"], inplace=True)

    # --- Calculate Hit Ratios ---
    # For walk-forward, "training hit ratio" isn't directly applicable in the same way,
    # as the model is continually retrained. We only assess performance on the 'test' set predictions.
    train_hit_ratio = np.nan  # N/A for this approach

    if not test.empty and "direction" in test.columns and "position" in test.columns:
        # Execution-aligned: position[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(
            test["direction"].iloc[1:].values, test["position"].iloc[:-1].values
        )
    else:
        test_hit_ratio = (
            np.nan
        )  # Cannot calculate if test set is empty or columns are missing

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print("\nModel Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio:.4f} (N/A for walk-forward SARIMAX)")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n" * 3)

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("Sample of test['position'] (first 10, last 10):")
        print(test["position"].head(10).to_string())
        print(test["position"].tail(10).to_string())
        print("=" * 50 + "\n" * 3)

    # --- Strategy Returns Calculation (shift position by 1 bar to avoid look-ahead bias) ---
    # Calculate the raw strategy returns
    test["strategy"] = test["position"].shift(1) * test["returns"]

    # --- Apply Transaction Costs ---
    # PTC is imported from algos.common.config
    transaction_cost_log_impact = np.log(1 - PTC)

    # Apply transaction cost only when the position changes.
    # `fillna(0)` handles the very first `.diff()` value, which is NaN.
    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff().fillna(0) != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    # --- Return Results ---
    # The run_backtest.py script will handle plotting and full risk analysis based on the returned test DataFrame.
    return test, model_fit
