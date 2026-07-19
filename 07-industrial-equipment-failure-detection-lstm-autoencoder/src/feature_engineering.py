from __future__ import annotations

import numpy as np
import pandas as pd


def sensor_summary(frame: pd.DataFrame, sensor_cols: list[str]) -> pd.DataFrame:
    """Return a compact sensor-quality and variability profile."""
    summary = frame[sensor_cols].describe().T
    summary["missing"] = frame[sensor_cols].isna().sum()
    summary["range"] = summary["max"] - summary["min"]
    return summary[["count", "mean", "std", "min", "max", "range", "missing"]]


def per_sensor_reconstruction_contribution(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """Mean absolute reconstruction error for each sensor feature."""
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed arrays must have identical shapes.")
    if original.ndim != 3:
        raise ValueError("Expected arrays with shape [samples, time_steps, features].")
    return np.mean(np.abs(original - reconstructed), axis=1)
