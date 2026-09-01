# # import numpy as np
# # import pandas as pd
# # from sklearn.cluster import KMeans
# # from sklearn.metrics import accuracy_score
# # from sklearn.preprocessing import StandardScaler
# # import sys # For sys.exit
# # import pickle # For saving/loading the model

# # # Import centralized utilities and configurations
# # from algos.common.utils import RedirectStdoutToFile
# # from algos.common.config import PTC # Import transaction cost percentage

# # def run_kmeans_strategy(data: pd.DataFrame, initial_train_split_ratio: float = 0.5, log_prefix: str = "KMeans_model"):
# #     """
# #     Implements a K-Means clustering-based trading strategy with a fixed train/test split.
# #     K-Means is used to cluster lagged returns, and clusters are mapped to trading signals (-1 or 1)
# #     based on the average direction within each cluster during the training phase.

# #     Args:
# #         data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
# #                              It should also implicitly contain 'annual_trading_periods' as an attribute
# #                              if needed for downstream calculations (though not directly used by this model's core logic).
# #         initial_train_split_ratio (float): Ratio for the initial train-test split.
# #         log_prefix (str): Prefix for the log file specific to this model.

# #     Returns:
# #         tuple: (test_df_with_predictions, final_model_object)
# #                test_df_with_predictions: DataFrame with 'predicted_returns' (cluster labels mapped to signals),
# #                                          'position', 'strategy', 'strategy_tc'.
# #                final_model_object: The last fitted KMeans model.
# #     """
# #     # --- Input Data Validation ---
# #     if data.empty:
# #         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #             print(f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting.")
# #         sys.exit(1)

# #     if 'returns' not in data.columns or 'direction' not in data.columns:
# #         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #             print(f"Fatal Error: Input data for {log_prefix} is missing 'returns' or 'direction' columns. Exiting.")
# #         sys.exit(1)

# #     # --- Feature Engineering: Create Lagged Returns ---
# #     lags = 5 # Number of lagged returns to use as features
# #     cols = [] # List to store column names of lagged features
# #     for lag in range(1, lags + 1):
# #         col = 'lag_{}'.format(lag)
# #         data[col] = data['returns'].shift(lag)
# #         cols.append(col)
# #     data.dropna(inplace=True) # Drop rows with NaN values introduced by shifting

# #     if data.empty:
# #         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #             print(f"Fatal Error: 'data' DataFrame is empty after creating lags and dropping NaNs for {log_prefix}. Exiting.")
# #         sys.exit(1)

# #     # --- Train/Test Split ---
# #     split = int(len(data) * initial_train_split_ratio)
# #     train = data.iloc[:split].copy()
# #     test = data.iloc[split:].copy()

# #     if train.empty or test.empty or len(cols) == 0:
# #         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #             print(f"Fatal Error: Insufficient data for training or testing {log_prefix}. Check data loading and preprocessing steps.")
# #         sys.exit(1)

# #     # --- Feature Scaling (Crucial for distance-based algorithms like KMeans) ---
# #     scaler = StandardScaler()
# #     X_train_scaled = scaler.fit_transform(train[cols])
# #     X_test_scaled = scaler.transform(test[cols])

# #     # --- KMeans Model Definition and Training ---
# #     # K-Means should ideally have n_clusters=2 for a binary trading decision (buy/sell).
# #     # random_state for reproducibility. 'n_init' set to 'auto' for modern sklearn versions.
# #     model = KMeans(n_clusters=2, random_state=0, n_init='auto')

# #     with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #         print('=' * 50)
# #         print(f"\nModel Training ({log_prefix}):\n")
# #         try:
# #             model.fit(X_train_scaled)
# #             print("KMeans centroids:\n", model.cluster_centers_)
# #             print("Training finished.")
# #         except Exception as e:
# #             print(f"An error occurred during {log_prefix} training: {e}")
# #             sys.exit(1) # Exit on critical training error
# #         print('=' * 50 + '\n')

# #     # --- Predict Cluster Labels ---
# #     train_cluster_labels = model.predict(X_train_scaled)
# #     test_cluster_labels = model.predict(X_test_scaled)

# #     # --- Map KMeans Clusters to Trading Signals (1 or -1) ---
# #     # Determine which cluster corresponds to a 'buy' (1) and which to a 'sell' (-1)
# #     # by looking at the average 'direction' within each cluster in the training set.
# #     cluster_to_direction_map = {}
# #     for cluster_id in np.unique(train_cluster_labels):
# #         directions_in_cluster = train['direction'][train_cluster_labels == cluster_id]
# #         if len(directions_in_cluster) > 0:
# #             # Assign the sign of the mean direction to the cluster.
# #             # If the mean direction is positive, assign 1 (buy); otherwise, assign -1 (sell).
# #             cluster_to_direction_map[cluster_id] = np.sign(directions_in_cluster.mean())
# #         else:
# #             # This case should be rare with default n_init and n_clusters=2
# #             with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #                 print(f"Warning: Cluster {cluster_id} is empty in the training set. Assigning default direction of 1.")
# #             cluster_to_direction_map[cluster_id] = 1 # Default to buy for empty clusters

# #     with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #         print("\nKMeans Cluster to Trading Direction Map:")
# #         print(cluster_to_direction_map)
# #         print('=' * 50 + '\n')

# #     # Apply the mapping to get trading positions
# #     # We assign the mapped binary prediction to 'predicted_returns' for consistency,
# #     # as KMeans is not predicting continuous returns directly.
# #     test['predicted_returns'] = np.array([cluster_to_direction_map[label] for label in test_cluster_labels])
# #     test['position'] = test['predicted_returns'] # 'position' is the direct trading signal

# #     # --- Calculate Hit Ratios ---
# #     # Convert training cluster labels to mapped directions for hit ratio calculation
# #     train_predictions_mapped = np.array([cluster_to_direction_map[label] for label in train_cluster_labels])
# #     train_hit_ratio = accuracy_score(train['direction'], train_predictions_mapped)

# #     # Ensure test['direction'] and test['position'] are valid before calculating accuracy
# #     if not test['direction'].empty and not test['position'].empty and len(test['direction']) == len(test['position']):
# #         test_hit_ratio = accuracy_score(test['direction'], test['position'])
# #     else:
# #         test_hit_ratio = np.nan # Cannot calculate if test set or predictions are problematic

# #     with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #         print('=' * 50)
# #         print("\nModel Performance (Hit Ratios):\n")
# #         print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
# #         print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
# #         print('=' * 50 + '\n' * 3)

# #     with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #         print("Sample of test['position'] (first 10, last 10):")
# #         print(test['position'].head(10).to_string())
# #         print(test['position'].tail(10).to_string())
# #         print('=' * 50 + '\n' * 3)

# #     # --- Strategy Returns Calculation ---
# #     test.dropna(subset=['position', 'returns'], inplace=True)
# #     if test.empty:
# #         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
# #             print(f"Error: 'test' DataFrame is empty after dropping NaNs for strategy calculation for {log_prefix}. Exiting.")
# #         sys.exit(1)

# #     test['strategy'] = test['position'] * test['returns']

# #     # --- Apply Transaction Costs ---
# #     # PTC is imported from algos.common.config
# #     transaction_cost_log_impact = np.log(1 - PTC)

# #     # Apply transaction cost only when position changes. fillna(0) handles the first row's diff().
# #     test['strategy_tc'] = np.where(test['position'].diff().fillna(0) != 0,
# #                                    test['strategy'] + transaction_cost_log_impact,
# #                                    test['strategy'])

# #     # --- Return Results ---
# #     # The run_backtest.py script will handle plotting and full risk analysis based on the returned test DataFrame.
# #     return test, model


# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# import sys # For sys.exit
# import pickle # For saving/loading the model

# # Import centralized utilities and configurations
# from algos.common.utils import RedirectStdoutToFile
# from algos.common.config import PTC # Import transaction cost percentage

# def run_kmeans_strategy(data: pd.DataFrame, initial_train_split_ratio: float = 0.5, log_prefix: str = "KMeans_model"):
#     """
#     Implements a K-Means clustering-based trading strategy with a fixed train/test split.
#     K-Means is used to cluster lagged returns, and clusters are mapped to trading signals (-1 or 1)
#     based on the average direction within each cluster during the training phase.

#     Args:
#         data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
#                              It should also implicitly contain 'annual_trading_periods' as an attribute
#                              if needed for downstream calculations (though not directly used by this model's core logic).
#         initial_train_split_ratio (float): Ratio for the initial train-test split.
#         log_prefix (str): Prefix for the log file specific to this model.

#     Returns:
#         tuple: (test_df_with_predictions, final_model_object)
#                test_df_with_predictions: DataFrame with 'predicted_returns' (cluster labels mapped to signals),
#                                          'position', 'strategy', 'strategy_tc'.
#                final_model_object: The last fitted KMeans model.
#     """
#     # --- Input Data Validation ---
#     if data.empty:
#         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#             print(f"Fatal Error: Input data is empty for {log_prefix} strategy. Exiting.")
#         sys.exit(1)

#     if 'returns' not in data.columns or 'direction' not in data.columns:
#         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#             print(f"Fatal Error: Input data for {log_prefix} is missing 'returns' or 'direction' columns. Exiting.")
#         sys.exit(1)

#     # --- Feature Engineering: Create Lagged Returns ---
#     lags = 4 # Number of lagged returns to use as features
#     cols = [] # List to store column names of lagged features
#     for lag in range(1, lags + 1):
#         col = 'lag_{}'.format(lag)
#         data[col] = data['returns'].shift(lag)
#         cols.append(col)
#     data.dropna(inplace=True) # Drop rows with NaN values introduced by shifting

#     if data.empty:
#         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#             print(f"Fatal Error: 'data' DataFrame is empty after creating lags and dropping NaNs for {log_prefix}. Exiting.")
#         sys.exit(1)

#     # --- Train/Test Split ---
#     split = int(len(data) * initial_train_split_ratio)
#     train = data.iloc[:split].copy()
#     test = data.iloc[split:].copy()

#     if train.empty or test.empty or len(cols) == 0:
#         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#             print(f"Fatal Error: Insufficient data for training or testing {log_prefix}. Check data loading and preprocessing steps.")
#         sys.exit(1)

#     # --- Feature Scaling (Crucial for distance-based algorithms like KMeans) ---
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(train[cols])
#     X_test_scaled = scaler.transform(test[cols])

#     # --- KMeans Model Definition and Training ---
#     model = KMeans(n_clusters=2, random_state=0, n_init='auto')

#     with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#         print('=' * 50)
#         print(f"\nModel Training ({log_prefix}):\n")
#         # NEW: Print class distribution in training set's actual directions
#         print("Training set 'direction' class distribution:")
#         print(train['direction'].value_counts(normalize=True))
#         print("-" * 50)
#         try:
#             model.fit(X_train_scaled)
#             print("KMeans centroids:\n", model.cluster_centers_)
#             print("Training finished.")
#             # NEW: Print distribution of cluster labels in training set
#             print("\nTraining set cluster label distribution:")
#             train_cluster_labels_initial = model.predict(X_train_scaled) # Need to predict after fit to get labels for current run
#             print(pd.Series(train_cluster_labels_initial).value_counts(normalize=True))
#         except Exception as e:
#             print(f"An error occurred during {log_prefix} training: {e}")
#             sys.exit(1)
#         print('=' * 50 + '\n')

#     # --- Predict Cluster Labels ---
#     train_cluster_labels = model.predict(X_train_scaled)
#     test_cluster_labels = model.predict(X_test_scaled)

#     # --- Map KMeans Clusters to Trading Signals (1 or -1) ---
#     cluster_to_direction_map = {}
#     for cluster_id in np.unique(train_cluster_labels):
#         directions_in_cluster = train['direction'][train_cluster_labels == cluster_id]
#         if len(directions_in_cluster) > 0:
#             cluster_to_direction_map[cluster_id] = np.sign(directions_in_cluster.mean())
#         else:
#             with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#                 print(f"Warning: Cluster {cluster_id} is empty in the training set. Assigning default direction of 1.")
#             cluster_to_direction_map[cluster_id] = 1

#     with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#         print("\nKMeans Cluster to Trading Direction Map:")
#         print(cluster_to_direction_map)
#         print('=' * 50 + '\n')

#     # Apply the mapping to get trading positions
#     test['predicted_returns'] = np.array([cluster_to_direction_map[label] for label in test_cluster_labels])
#     test['position'] = test['predicted_returns']

#     # --- Calculate Hit Ratios ---
#     train_predictions_mapped = np.array([cluster_to_direction_map[label] for label in train_cluster_labels])
#     train_hit_ratio = accuracy_score(train['direction'], train_predictions_mapped)

#     if not test['direction'].empty and not test['position'].empty and len(test['direction']) == len(test['position']):
#         test_hit_ratio = accuracy_score(test['direction'], test['position'])
#     else:
#         test_hit_ratio = np.nan

#     with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#         print('=' * 50)
#         print("\nModel Performance (Hit Ratios):\n")
#         print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
#         print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
#         print('=' * 50 + '\n')

#         # NEW: Print distribution of the final mapped 'position' (predictions) in test set
#         print("Test set 'position' (predictions) distribution:")
#         if 'position' in test.columns and not test['position'].empty:
#             print(test['position'].value_counts(normalize=True))
#         else:
#             print("Test 'position' column is empty or not created.")
#         print("-" * 50)

#         print("Sample of test['position'] (first 10, last 10):")
#         print(test['position'].head(10).to_string())
#         print(test['position'].tail(10).to_string())
#         print('=' * 50 + '\n' * 3)

#     # --- Strategy Returns Calculation ---
#     test.dropna(subset=['position', 'returns'], inplace=True)
#     if test.empty:
#         with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
#             print(f"Error: 'test' DataFrame is empty after dropping NaNs for strategy calculation for {log_prefix}. Exiting.")
#         sys.exit(1)

#     test['strategy'] = test['position'] * test['returns']

#     # --- Apply Transaction Costs ---
#     transaction_cost_log_impact = np.log(1 - PTC)

#     test['strategy_tc'] = np.where(test['position'].diff().fillna(0) != 0,
#                                    test['strategy'] + transaction_cost_log_impact,
#                                    test['strategy'])

#     return test, model

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys  # For sys.exit
import pickle  # For saving/loading the model

# Import centralized utilities and configurations
from algos.common.utils import RedirectStdoutToFile
from algos.common.config import PTC  # Import transaction cost percentage
from algos.common.embargo_utils import (
    calculate_embargo_size,
    validate_embargo_split,
    get_embargo_message,
)


def run_kmeans_strategy(
    data: pd.DataFrame,
    initial_train_split_ratio: float = 0.5,
    log_prefix: str = "KMeans_model",
    embargo_pct: float = 0.02,
    **kwargs,
):
    """
    Implements a K-Means clustering-based trading strategy with a fixed train/test split.
    K-Means is used to cluster lagged returns, and clusters are mapped to trading signals (-1, 0, or 1)
    based on the average direction within each cluster during the training phase.

    Args:
        data (pd.DataFrame): The preprocessed DataFrame with 'returns' and 'direction' columns.
                             It should also implicitly contain 'annual_trading_periods' as an attribute
                             if needed for downstream calculations (though not directly used by this model's core logic).
        initial_train_split_ratio (float): Ratio for the initial train-test split.
        log_prefix (str): Prefix for the log file specific to this model.

    Returns:
        tuple: (test_df_with_predictions, final_model_object)
               test_df_with_predictions: DataFrame with 'predicted_returns' (cluster labels mapped to signals),
                                         'position', 'strategy', 'strategy_tc'.
               final_model_object: The last fitted KMeans model.
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

    # --- Feature Engineering: Create Lagged Returns ---
    lags = 3  # Number of lagged returns to use as features
    cols = []  # List to store column names of lagged features
    for lag in range(1, lags + 1):
        col = "lag_{}".format(lag)
        data[col] = data["returns"].shift(lag)
        cols.append(col)
    data.dropna(inplace=True)  # Drop rows with NaN values introduced by shifting

    if data.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: 'data' DataFrame is empty after creating lags and dropping NaNs for {log_prefix}. Exiting."
            )
        sys.exit(1)

    # --- Train/Test Split with Embargo ---
    split = int(len(data) * initial_train_split_ratio)

    if embargo_pct > 0:
        embargo_size = calculate_embargo_size(len(data), embargo_pct, min_samples=5)
        train = data.iloc[:split].copy()
        test = data.iloc[split + embargo_size :].copy()
        validate_embargo_split(len(train), embargo_size, len(test), min_test_samples=30)
        print(get_embargo_message(embargo_size, len(data), len(train), len(test)))
    else:
        train = data.iloc[:split].copy()
        test = data.iloc[split:].copy()
        print(
            "\nNote: Embargo disabled (embargo_pct=0). Using simple train/test split."
        )

    if train.empty or test.empty or len(cols) == 0:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Fatal Error: Insufficient data for training or testing {log_prefix}. Check data loading and preprocessing steps."
            )
        sys.exit(1)

    # --- Feature Scaling (Crucial for distance-based algorithms like KMeans) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[cols])
    X_test_scaled = scaler.transform(test[cols])

    # --- KMeans Model Definition and Training ---
    # Changed n_clusters to 3 to allow for a neutral position
    model = KMeans(n_clusters=3, random_state=0, n_init="auto")

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print(f"\nModel Training ({log_prefix}):\n")
        print("Training set 'direction' class distribution:")
        print(train["direction"].value_counts(normalize=True))
        print("-" * 50)
        try:
            model.fit(X_train_scaled)
            print("KMeans centroids:\n", model.cluster_centers_)
            print("Training finished.")
            train_cluster_labels_initial = model.predict(X_train_scaled)
            print("\nTraining set cluster label distribution:")
            print(pd.Series(train_cluster_labels_initial).value_counts(normalize=True))
        except Exception as e:
            print(f"An error occurred during {log_prefix} training: {e}")
            sys.exit(1)
        print("=" * 50 + "\n")

    # --- Predict Cluster Labels ---
    train_cluster_labels = model.predict(X_train_scaled)
    test_cluster_labels = model.predict(X_test_scaled)

    # --- Map KMeans Clusters to Trading Signals (1, -1, or 0) ---
    cluster_mean_directions = {}
    for cluster_id in np.unique(train_cluster_labels):
        directions_in_cluster = train["direction"][train_cluster_labels == cluster_id]
        if len(directions_in_cluster) > 0:
            cluster_mean_directions[cluster_id] = directions_in_cluster.mean()
        else:
            with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
                print(
                    f"Warning: Cluster {cluster_id} is empty in the training set. Assigning default mean of 0."
                )
            cluster_mean_directions[cluster_id] = 0.0  # Assign 0 if cluster is empty

    # Sort clusters by their mean direction to assign buy, sell, or cash
    sorted_clusters = sorted(cluster_mean_directions.items(), key=lambda item: item[1])

    cluster_to_direction_map = {}
    # Map the cluster with the lowest mean direction to -1 (sell)
    cluster_to_direction_map[sorted_clusters[0][0]] = -1.0
    # Map the middle cluster to 0 (cash/neutral)
    cluster_to_direction_map[sorted_clusters[1][0]] = 0.0
    # Map the cluster with the highest mean direction to 1 (buy)
    cluster_to_direction_map[sorted_clusters[2][0]] = 1.0

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("\nKMeans Cluster to Trading Direction Map:")
        print(cluster_to_direction_map)
        print("=" * 50 + "\n")

    # Apply the mapping to get trading positions
    test["predicted_returns"] = np.array(
        [cluster_to_direction_map[label] for label in test_cluster_labels]
    )
    test["position"] = test[
        "predicted_returns"
    ]  # 'position' is the direct trading signal

    # --- Calculate Hit Ratios ---
    train_predictions_mapped = np.array(
        [cluster_to_direction_map[label] for label in train_cluster_labels]
    )
    # Note: Accuracy is still calculated against the original direction (-1, 1).
    # Predictions of 0 will be counted as incorrect here, which is appropriate for hit ratio.
    train_hit_ratio = accuracy_score(
        train["direction"], np.sign(train_predictions_mapped)
    )  # Use np.sign for 0 in hit ratio

    if (
        not test["direction"].empty
        and not test["position"].empty
        and len(test["direction"]) == len(test["position"])
    ):
        # Execution-aligned: position[t] predicts direction[t+1]
        test_hit_ratio = accuracy_score(
            test["direction"].iloc[1:].values,
            np.sign(test["position"].iloc[:-1].values),
        )
    else:
        test_hit_ratio = np.nan

    with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
        print("=" * 50)
        print("\nModel Performance (Hit Ratios):\n")
        print(f"Train Hit Ratio: {train_hit_ratio:.4f}")
        print(f"Test Hit Ratio: {test_hit_ratio:.4f}")
        print("=" * 50 + "\n")

        print("Test set 'position' (predictions) distribution:")
        if "position" in test.columns and not test["position"].empty:
            print(test["position"].value_counts(normalize=True))
        else:
            print("Test 'position' column is empty or not created.")
        print("-" * 50)

        print("Sample of test['position'] (first 10, last 10):")
        print(test["position"].head(10).to_string())
        print(test["position"].tail(10).to_string())
        print("=" * 50 + "\n" * 3)

    # --- Strategy Returns Calculation ---
    test.dropna(subset=["position", "returns"], inplace=True)
    if test.empty:
        with RedirectStdoutToFile(f"{log_prefix}_output.txt"):
            print(
                f"Error: 'test' DataFrame is empty after dropping NaNs for strategy calculation for {log_prefix}. Exiting."
            )
        sys.exit(1)

    test["strategy"] = test["position"].shift(1) * test["returns"]

    # --- Apply Transaction Costs ---
    transaction_cost_log_impact = np.log(1 - PTC)

    test["strategy_tc"] = np.where(
        test["position"].shift(1).diff().fillna(0) != 0,
        test["strategy"] + transaction_cost_log_impact,
        test["strategy"],
    )
    # Remove first row NaN from position shift
    test.dropna(subset=["strategy"], inplace=True)

    return test, model
