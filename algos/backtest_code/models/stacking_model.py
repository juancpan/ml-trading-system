import numpy as np
import pandas as pd
import sys  # For sys.exit
import pickle  # For saving/loading models

# Conditional imports for Stacking Models
_HAS_STACKING_LIBS = False
try:
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from xgboost import XGBClassifier  # New base model
    from sklearn.ensemble import (
        StackingClassifier,
    )  # Still relevant for conceptual understanding, but we're doing manual walk-forward
    from sklearn.linear_model import LogisticRegression  # Meta-model
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    _HAS_STACKING_LIBS = True
except ImportError as e:
    print(
        f"Warning: Missing required libraries for model stacking (scikit-learn or xgboost). Stacking model functionality will be skipped. Error: {e}",
        file=sys.stderr,
    )

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage


def run_stacking_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    lags: int = 5,
    log_prefix: str = "Models_Stacking",
    **kwargs,
):
    """
    Implements a stacking ensemble strategy with walk-forward validation.
    Base models include Gaussian Naive Bayes, Support Vector Classifier, and XGBoost Classifier.
    A Logistic Regression model acts as the meta-learner to combine their predictions.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'price' columns.
        initial_train_split_ratio (float): Ratio for the initial training data size before walk-forward.
        lags (int): Number of lagged returns to use as features for base models.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'position' (trading signals),
                                         'strategy', 'strategy_tc'.
               final_model_object: The last trained LogisticRegression meta-model object.
                                   Returns None if required libraries are missing.
    """
    # --- Check for Stacking library availability early ---
    if not _HAS_STACKING_LIBS:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Required stacking libraries not found. Cannot run {log_prefix} strategy. Exiting."
            )
        return pd.DataFrame(), None  # Return empty DataFrame and None model

    # --- Input Data Validation ---
    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting."
            )
        sys.exit(1)

    if "returns" not in data.columns or "price" not in data.columns:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Input data for {log_prefix} is missing 'returns' or 'price' columns. Exiting."
            )
        sys.exit(1)

    # --- Feature Engineering ---
    feature_cols = []
    for lag in range(1, lags + 1):
        col = f"lag_{lag}"
        data[col] = data["returns"].shift(lag)
        feature_cols.append(col)

    data.dropna(inplace=True)

    # Ensure direction uses {-1, 1} encoding (short/long).
    # Preserves carry-adjusted direction from data_loader for forex tickers.
    if "direction" not in data.columns:
        data["direction"] = np.where(data["returns"] > 0, 1, -1)
    else:
        pass  # Upstream direction already uses {-1, 1} encoding

    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: 'data' DataFrame is empty after creating features and dropping NaNs for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Train/Test Split ---
    split = int(len(data) * initial_train_split_ratio)
    if split == 0 or split >= len(data):
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Invalid split ratio or insufficient data for {log_prefix}. Split results in empty train/test. Exiting."
            )
        sys.exit(1)

    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    if (
        train.empty
        or test.empty
        or not feature_cols
        or not all(col in train.columns for col in feature_cols)
    ):
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Insufficient data for stacking model training or testing, or missing feature columns for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Walk-Forward Validation for Stacking ---
    # Initialize lists to store final stacking predictions for the test set
    stacking_predictions = []

    # Initial history for walk-forward
    current_X_train = train[feature_cols].to_numpy()
    current_y_train = train["direction"].to_numpy()
    current_test_indices = test.index  # Keep track of original test indices

    # Accumulators for meta-model training data (grow with each step)
    meta_X_train_accumulator = []  # Stores rows of base model predictions
    meta_y_train_accumulator = []  # Stores corresponding actual directions

    final_meta_model = None  # To store the last fitted meta-model

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\n{log_prefix} Model Training and Forecasting (Walk-Forward):\n")
        print("Training base models and collecting predictions for meta-model...")

        for t in range(len(test)):
            # Data for the current single test observation
            X_test_observation = test[feature_cols].iloc[t : t + 1].to_numpy()
            y_test_observation = test["direction"].iloc[
                t
            ]  # Actual direction for current test observation

            # Initialize a new StandardScaler for each step to avoid data leakage
            # It fits on current_X_train and transforms both training and current test observation
            scaler = StandardScaler()
            current_X_train_scaled = scaler.fit_transform(current_X_train)
            X_test_observation_scaled = scaler.transform(X_test_observation)

            # --- Collect Base Model Predictions (for current observation) ---
            current_meta_features_row = []  # This will be the single row of meta-features for time t

            # GNB Base Model
            try:
                gnb_model = GaussianNB()
                gnb_model.fit(current_X_train_scaled, current_y_train)
                gnb_pred_proba = gnb_model.predict_proba(X_test_observation_scaled)[
                    :, 1
                ][0]
                current_meta_features_row.append(gnb_pred_proba)
            except Exception as e:
                current_meta_features_row.append(np.nan)
                print(
                    f"Warning: GaussianNB fit/predict failed at step {t} with error: {e}"
                )

            # SVC Base Model
            try:
                svc_model = SVC(kernel="linear", probability=True, random_state=1000)
                svc_model.fit(current_X_train_scaled, current_y_train)
                svc_pred_proba = svc_model.predict_proba(X_test_observation_scaled)[
                    :, 1
                ][0]
                current_meta_features_row.append(svc_pred_proba)
            except Exception as e:
                svc_meta_features.append(np.nan)
                print(f"Warning: SVC fit/predict failed at step {t} with error: {e}")

            # XGBoost Base Model
            try:
                xgb_model = XGBClassifier(
                    use_label_encoder=False, eval_metric="logloss", random_state=1000
                )
                xgb_model.fit(current_X_train_scaled, current_y_train)
                xgb_pred_proba = xgb_model.predict_proba(X_test_observation_scaled)[
                    :, 1
                ][0]
                current_meta_features_row.append(xgb_pred_proba)
            except Exception as e:
                current_meta_features_row.append(np.nan)
                print(
                    f"Warning: XGBoost fit/predict failed at step {t} with error: {e}"
                )

            # --- Update Meta-Model's Training Data Accumulators ---
            # These lists will grow with each step, effectively providing a growing window for the meta-model
            # Note: We append the actual target (y_test_observation) for the meta-model training.
            meta_X_train_accumulator.append(current_meta_features_row)
            meta_y_train_accumulator.append(y_test_observation)

            # Convert accumulators to numpy arrays for meta-model training (handle NaNs)
            # This is the data the meta-model will train on *up to this point in time t-1*
            temp_meta_X_train = np.array(meta_X_train_accumulator)
            temp_meta_y_train = np.array(meta_y_train_accumulator)

            # Drop rows with NaNs from meta-features (where base models failed to predict)
            nan_rows = np.any(np.isnan(temp_meta_X_train), axis=1)
            X_meta_train_cleaned = temp_meta_X_train[~nan_rows]
            y_meta_train_cleaned = temp_meta_y_train[~nan_rows]

            # --- Train and Predict with Meta-Model (for current step 't') ---
            current_stacking_prediction = (
                np.nan
            )  # Default if meta-model training fails or no valid data

            if X_meta_train_cleaned.shape[0] > 0 and not np.any(
                np.isnan(current_meta_features_row)
            ):
                try:
                    # Meta-model is Logistic Regression
                    meta_model_lr = LogisticRegression(
                        random_state=1000, solver="liblinear"
                    )
                    meta_model_lr.fit(X_meta_train_cleaned, y_meta_train_cleaned)
                    final_meta_model = (
                        meta_model_lr  # Keep track of the *last* fitted meta-model
                    )

                    # Predict using the *currently trained* meta-model on the *current single meta-feature row*
                    # Reshape for single sample prediction: (1, num_features)
                    current_stacking_prediction = meta_model_lr.predict(
                        np.array(current_meta_features_row).reshape(1, -1)
                    )[0]
                except Exception as e:
                    print(
                        f"Warning: Meta-model fit/predict failed at step {t} with error: {e}. Prediction for this step will be NaN."
                    )
            else:
                print(
                    f"Warning: No valid meta-features to train meta-model at step {t} or current meta-features are NaN. Stacking prediction will be NaN."
                )

            stacking_predictions.append(current_stacking_prediction)

            # --- Update historical data for the next iteration (growing window) ---
            # Append the current test observation to the training data for the next step
            current_X_train = np.vstack([current_X_train, X_test_observation])
            current_y_train = np.append(current_y_train, y_test_observation)

            if (t + 1) % 100 == 0 or t == len(test) - 1:
                print(
                    f"Processed {t + 1}/{len(test)} test points for {log_prefix} (training and forecasting stacking)."
                )

        print(f"\n{log_prefix} Walk-Forward Forecasting Complete.")
        print("=" * 50 + "\n")

    # --- Assign final stacking predictions to test DataFrame ---
    test["predicted_direction"] = pd.Series(
        stacking_predictions, index=current_test_indices
    )
    test.dropna(
        subset=["predicted_direction"], inplace=True
    )  # Remove rows where prediction was not made due to NaNs

    # Predictions are already {-1, 1}; use directly as position
    test["position"] = np.where(test["predicted_direction"] == 1, 1, -1)

    test.dropna(
        subset=["direction"], inplace=True
    )  # Clean 'direction' if any NaNs arose

    # --- Calculate Hit Ratios ---
    train_hit_ratio = (
        np.nan
    )  # Not applicable directly for this walk-forward stacking approach

    if not test.empty and "direction" in test.columns and "position" in test.columns:
        # Execution-aligned: position[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(
            test["direction"].iloc[1:].values, test["position"].iloc[:-1].values
        )
    else:
        test_hit_ratio = np.nan

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print("\nModel Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio} (N/A for walk-forward stacking)")
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
    transaction_cost_log_impact = np.log(
        1 - PTC
    )  # PTC is imported from algos.common.config

    position_shifted = test["position"].shift(1)
    test["strategy_tc"] = np.where(
        position_shifted.diff().fillna(0) != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    # --- Return Results ---
    return test, final_meta_model
