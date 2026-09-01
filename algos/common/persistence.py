# algos/common/persistence.py

import pickle
import tensorflow as tf  # Assuming you might save Keras/TF models
import os
from pathlib import Path
import sys
import json

# ---------- version helpers --------------------------------------------------
# Collect library versions at save-time so that load-time can warn early about
# incompatibilities (sklearn is particularly strict about cross-version pickle).


def _collect_version_info() -> dict:
    """Return a dict of key library versions present in the current env."""
    info = {"python": sys.version}
    for lib_name in ("sklearn", "statsmodels", "xgboost", "tensorflow", "numpy"):
        try:
            mod = __import__(lib_name)
            info[lib_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return info


# --- Define and create necessary directories relative to project root
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
        # print("Warning: Could not automatically detect project root. Using current working directory as base.", file=sys.stderr)
        project_root_dir = current_execution_dir
except NameError:
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root_dir = (
        current_script_dir.parent.parent
    )  # Adjust based on persistence.py's location

model_dumps_dir = project_root_dir / "algos" / "model_dumps"
logs_dir = project_root_dir / "logs"

for d in [model_dumps_dir, logs_dir]:
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)


class RedirectStdoutToFile:
    def __init__(self, filename="output.txt", mode="a"):
        self.filename = logs_dir / filename
        self.mode = mode
        self.original_stdout = sys.stdout
        self.file = None

    def __enter__(self):
        # Check the global suppress flag from utils.py
        try:
            from algos.common.utils import _SUPPRESS_FILE_LOGS

            if _SUPPRESS_FILE_LOGS:
                self.file = open(os.devnull, "w")
                sys.stdout = self.file
                return self
        except ImportError:
            pass
        self.file = open(self.filename, self.mode)
        sys.stdout = self.file
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()


def _save_keras_with_retry(model_obj, save_path: Path, max_retries: int = 3):
    """
    Save a Keras model with retry logic for HDF5 file lock contention.

    When multiple processes save .keras files concurrently, h5py can throw
    BlockingIOError (errno 35) or OSError ("file signature not found") due
    to HDF5's global file locking. Retrying with a brief delay resolves this.
    """
    import time
    import random

    for attempt in range(max_retries):
        try:
            model_obj.save(save_path)
            return  # Success
        except (BlockingIOError, OSError) as e:
            if attempt < max_retries - 1:
                # Brief random backoff to desynchronize concurrent saves
                delay = 0.5 + random.random() * 1.5  # 0.5-2.0 seconds
                print(
                    f"  HDF5 save contention (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {delay:.1f}s: {e}"
                )
                time.sleep(delay)
                # Clean up corrupted file if it exists
                if save_path.exists():
                    try:
                        save_path.unlink()
                    except OSError:
                        pass
            else:
                raise  # Final attempt failed, propagate


def save_model(
    model_obj,
    model_name: str,
    ticker: str,
    symbol: str,
    start: str,
    end: str,
    interval: str,
    timestamp: str,
):
    """
    Saves the trained model object to a file. Handles different model types.

    Args:
        model_obj: The trained model object (e.g., Keras model, scikit-learn model).
        model_name (str): The name of the model (e.g., 'arima', 'cnn', 'dqn', 'sklearn_dnn').
        ticker (str): The financial instrument ticker.
        symbol (str): The price symbol used (e.g., 'Adj Close').
        start (str): Start date of the data.
        end (str): End date of the data.
        interval (str): Data interval.
        timestamp (str): A unique timestamp for file naming.
    """
    filename_base = (
        f"{model_name}_algorithm_{ticker}_{symbol}_{start}_{end}_{interval}_{timestamp}"
    )

    with RedirectStdoutToFile(
        f"{model_name.upper()}_model_save_output_{timestamp}.txt"
    ):
        try:
            # Check if it's a Keras model (assuming TensorFlow backend)
            if isinstance(model_obj, tf.keras.Model):
                model_save_path = model_dumps_dir / f"{filename_base}.keras"
                # Retry Keras saves to handle HDF5 file lock contention
                # when multiple processes save concurrently.
                _save_keras_with_retry(model_obj, model_save_path, max_retries=3)
                print(f"\nSaved Keras model to {model_save_path.resolve()}")
            else:  # Assume it's a scikit-learn model or other pickle-able object
                model_save_path = model_dumps_dir / f"{filename_base}.pkl"
                # Use protocol=4 explicitly for maximum cross-version
                # compatibility (supported by Python >= 3.4).  The default
                # protocol varies by Python version and can create files
                # unreadable on older interpreters.
                with open(model_save_path, "wb") as f:
                    pickle.dump(model_obj, f, protocol=4)
                print(
                    f"\nSaved {type(model_obj).__name__} model to {model_save_path.resolve()}"
                )

            # Save version metadata alongside the model for diagnostics.
            # This lets the loading side warn early about mismatches.
            version_path = (
                model_save_path.parent / f"{model_save_path.stem}_versions.json"
            )
            try:
                with open(version_path, "w") as vf:
                    json.dump(_collect_version_info(), vf, indent=2)
            except Exception:
                pass  # best-effort; don't fail the main save

        except Exception as e:
            print(f"\nError saving {model_name} model: {e}")


def save_arima_settings(ticker: str, settings: dict, timestamp: str):
    """
    Save ARIMA-specific settings alongside the model.

    Args:
        ticker (str): The financial instrument ticker.
        settings (dict): ARIMA settings including signal_method, thresholds, etc.
        timestamp (str): A unique timestamp for file naming (should match model timestamp).
    """
    filename = f"arima_settings_{ticker}_{timestamp}.json"
    settings_path = model_dumps_dir / filename

    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"Saved ARIMA settings to {settings_path.resolve()}")
        return settings_path
    except Exception as e:
        print(f"Error saving ARIMA settings: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_arima_settings(ticker: str, timestamp: str = None):
    """
    Load ARIMA settings for a given ticker.

    Args:
        ticker (str): The financial instrument ticker.
        timestamp (str): Optional timestamp to load specific settings.
                        If None, loads the most recent settings.

    Returns:
        dict: ARIMA settings or None if not found.
    """
    try:
        if timestamp:
            # Load specific settings file
            filename = f"arima_settings_{ticker}_{timestamp}.json"
            settings_path = model_dumps_dir / filename
            if settings_path.exists():
                with open(settings_path, "r") as f:
                    return json.load(f)
        else:
            # Find the most recent settings file for this ticker
            pattern = f"arima_settings_{ticker}_*.json"
            settings_files = list(model_dumps_dir.glob(pattern))

            if settings_files:
                # Sort by modification time and get the most recent
                most_recent = sorted(settings_files, key=lambda x: x.stat().st_mtime)[
                    -1
                ]
                with open(most_recent, "r") as f:
                    return json.load(f)

        return None
    except Exception as e:
        print(f"Error loading ARIMA settings: {e}")
        return None
