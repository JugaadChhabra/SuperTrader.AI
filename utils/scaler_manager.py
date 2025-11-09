"""Centralized scaler manager for SuperTrader.AI

Provides functions to load and save feature scalers using the project's
configuration. This removes hardcoded paths across the codebase.
"""
from pathlib import Path
import joblib
import pickle
from typing import Optional

try:
    from configs.config import get_config
except Exception:
    # Fallback: relative import when running scripts directly
    from ..configs.config import get_config


def get_scaler_path() -> str:
    """Return the configured scaler path as a string."""
    cfg = get_config()
    paths = cfg.get_model_paths()
    scaler = paths.get('feature_scaler')
    return scaler


def load_scaler(path: Optional[str] = None):
    """Load a scaler object (joblib/pickle) from configured path or provided path.

    Returns the scaler object or None if not found.
    """
    scaler_path = Path(path or get_scaler_path())
    if not scaler_path.exists():
        return None

    # Try joblib first, then pickle
    try:
        return joblib.load(scaler_path)
    except Exception:
        try:
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None


def save_scaler(scaler, path: Optional[str] = None) -> str:
    """Save scaler to path (joblib) and return path used."""
    scaler_path = Path(path or get_scaler_path())
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return str(scaler_path)
