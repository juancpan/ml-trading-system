import numpy as np
import pandas as pd
import sys  # For sys.exit
import pickle  # For saving/loading the statsmodels VAR model

# --- Conditional imports for VAR
_HAS_VAR_LIBS = False
try:
    from statsmodels.tsa.api import VAR  # Correct import for VAR model

    _HAS_VAR_LIBS = True
except ImportError as e:
    print(
        f"Warning: Missing required statsmodels library for VAR. VAR model functionality will be skipped. Error: {e}",
        file=sys.stderr,
    )

from sklearn.metrics import accuracy_score

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage
from algos.common.embargo_utils import (
    apply_embargo_to_walk_forward,
    get_embargo_message,
)


def run_var_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    var_order: int = 5,
    log_prefix: str = "VAR_model",
    embargo_pct: float = 0.02,
    interval: str = "1d",
    **kwargs,
):
    """
    Implements a Vector Autoregression (VAR) model-based trading strategy with embargo.
    The VAR model forecasts the next period's return (among other variables),
    and this forecast is then used to generate trading signals.
    It uses a walk-forward validation approach without explicit retraining intervals,
    re-fitting at each step.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
                             It should also contain any other series to be included in the VAR model.
        initial_train_split_ratio (float): Ratio for the initial training data size before walk-forward.
        var_order (int): The number of lags (p) to include in the VAR(p) model.
        log_prefix (str): Prefix for the log file specific to this model.
        embargo_pct (float): Embargo percentage (default 2%). Set to 0 to disable.
        interval (str): Data interval for auto-adjusting minimums ('1d', '1wk', '1mo').
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'position' (trading signals),
                                         'strategy', 'strategy_tc'.
               final_model_object: The last fitted statsmodels VARResultsWrapper object. Returns None if VAR libs are missing.
    """
    # --- Check for VAR library availability early ---
    if not _HAS_VAR_LIBS:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: statsmodels VAR library not found. Cannot run {log_prefix} strategy. Exiting."
            )
        return pd.DataFrame(), None  # Return empty DataFrame and None model

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

    # --- Feature Engineering: Prepare VAR Endogenous Variables ---
    # For VAR, we need all series that are part of the multivariate system.
    # Here, we'll use 'returns' and its own lags as endogenous variables.
    # You could extend this to include other relevant series (e.g., 'volume_returns', 'volatility')
    # if they are added to the 'data' DataFrame and you want to model their interdependencies.
    var_endog_cols = ["returns"]
    for lag in range(1, var_order + 1):
        # We'll use the shifted 'returns' as other endogenous variables in the VAR system.
        # Note: This means 'returns' at t-1, t-2, etc., are treated as separate time series in the VAR.
        # If your intention was to use these as exogenous features for predicting 'returns' only,
        # an AR(I)MA or a simple regression model might be more appropriate.
        # For a true VAR, you'd typically have multiple *different* time series that are all endogenous.
        # For now, we'll proceed as if these lagged returns are separate endogenous series.
        col = f"lag_{lag}"
        data[col] = data["returns"].shift(lag)
        var_endog_cols.append(col)

    data.dropna(inplace=True)  # Drop rows with NaN values introduced by shifting

    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: 'data' DataFrame is empty after creating VAR features and dropping NaNs for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Train/Test Split with Embargo ---
    split = int(len(data) * initial_train_split_ratio)
    if split == 0 or split >= len(data):
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Invalid split ratio or insufficient data for {log_prefix}. Split results in empty train/test. Exiting."
            )
        sys.exit(1)

    # Initial split
    train_initial = data[var_endog_cols].iloc[:split].copy()
    test_initial = data[var_endog_cols].iloc[split:].copy()

    # Apply embargo (interval-aware)
    train_data_for_var, test_data_for_var = apply_embargo_to_walk_forward(
        train_initial, test_initial, embargo_pct=embargo_pct, interval=interval
    )

    # Log embargo if enabled
    if embargo_pct > 0:
        embargo_size = len(test_initial) - len(test_data_for_var)
        print(
            get_embargo_message(
                embargo_size, len(data), len(train_data_for_var), len(test_data_for_var)
            )
        )

    # Create the 'test' DataFrame that will accumulate predictions and strategy returns
    # Must align with embargoged test_data_for_var indices
    if embargo_pct > 0:
        embargo_size = len(test_initial) - len(test_data_for_var)
        test = data.iloc[split + embargo_size :].copy()  # Skip embargo samples
    else:
        test = data.iloc[split:].copy()  # Original behavior

    if train_data_for_var.empty or test_data_for_var.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Train or test set for VAR model is empty after splitting for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # Initial history for the VAR model
    history_var_data = train_data_for_var  # Start with the training data

    predictions_list = []
    final_model_fit = None  # To store the last fitted VAR model object for persistence

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Training and Forecasting (Walk-Forward):\n")

        # The VAR model is re-fitted at each step in this walk-forward approach.
        # This can be computationally intensive for large datasets or frequent steps.
        for t in range(len(test_data_for_var)):
            current_history_array = history_var_data.values  # VAR expects numpy array

            try:
                # Statsmodels VAR requires at least 'var_order' observations for fitting
                # For robust estimation, significantly more observations are recommended.
                if current_history_array.shape[0] < var_order:
                    print(
                        f"Warning: Insufficient history ({current_history_array.shape[0]} samples) for VAR fit at step {t}. Need at least {var_order} observations. Appending NaN."
                    )
                    predictions_list.append(np.nan)
                    # No model to keep if insufficient history for current step
                    final_model_fit = None
                else:
                    # Fit VAR model on the current history
                    model = VAR(current_history_array)
                    # maxlags=var_order ensures the specified lag order is used.
                    # ic='aic' can select the best lag order up to maxlags if you want it to be determined automatically,
                    # but here we explicitly set maxlags and often, for VAR(p), you'd just do model.fit(var_order).
                    model_fit_current = model.fit(
                        maxlags=var_order, ic=None, trend="c"
                    )  # 'c' for constant, 'nc' for no constant

                    # Forecast 'steps=1' into the future
                    # The `forecast` method requires `y` which is the array of the most recent `var_order` observations.
                    lag_input = current_history_array[-var_order:]
                    # The forecast returns a 2D array: (steps, number_of_endogenous_variables)
                    # We need the prediction for 'returns', which is the first variable in our `var_endog_cols` list.
                    # So, it's [0, 0] for the first step and the first variable.
                    yhat = model_fit_current.forecast(y=lag_input, steps=1)[
                        0, var_endog_cols.index("returns")
                    ]
                    predictions_list.append(yhat)

                    # Store the last successfully fitted model for persistence
                    if t == len(test_data_for_var) - 1:
                        final_model_fit = model_fit_current

            except Exception as e:
                print(
                    f"Warning: VAR model fit/forecast failed at step {t} with error: {e}. Appending NaN."
                )
                predictions_list.append(np.nan)
                final_model_fit = None  # Clear model if failure at current step

            # Add the actual observation from the test set to the history for the next iteration
            # Ensure the new observation has the same columns as history_var_data
            current_observation = data[var_endog_cols].iloc[[split + t]]
            history_var_data = pd.concat([history_var_data, current_observation])

        # Align predictions with the test DataFrame.
        # Use the original index of the test set to align predictions.
        test["predicted_returns"] = pd.Series(
            predictions_list, index=test.index[: len(predictions_list)]
        )
        test.dropna(
            subset=["predicted_returns"], inplace=True
        )  # Drop rows where prediction failed or was NaN

        print("=" * 50 + "\n")

    # --- Convert Predicted Returns to Trading Signals (Position) ---
    # Using np.sign for consistency across all models
    test["position"] = np.sign(test["predicted_returns"].values).flatten()
    # Handle edge case where sign might return 0 (for exactly 0 prediction)
    test["position"] = np.where(test["position"] == 0, 1, test["position"])

    # Ensure 'direction' column is correctly defined for accuracy_score
    test.dropna(
        subset=["returns", "position"], inplace=True
    )  # Ensure data integrity after predictions

    # --- Calculate Hit Ratios ---
    train_hit_ratio = (
        np.nan
    )  # Not directly applicable in this walk-forward VAR prediction context

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
        print(f"Train Hit Ratio: {train_hit_ratio} (N/A for walk-forward VAR)")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n" * 3)

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("Sample of test['position'] (first 10, last 10):")
        print(test["position"].head(10).to_string())
        print(test["position"].tail(10).to_string())
        print("=" * 50 + "\n" * 3)

    # --- Strategy Returns Calculation (shift position by 1 bar to avoid look-ahead bias) ---
    test["strategy"] = test["position"].shift(1) * test["returns"]

    # --- Apply Transaction Costs ---
    # PTC is imported from algos.common.config
    transaction_cost_log_impact = np.log(1 - PTC)

    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff().fillna(0) != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    # --- Return Results ---
    # The run_backtest.py script will handle plotting and full risk analysis based on the returned test DataFrame.
    return test, final_model_fit
