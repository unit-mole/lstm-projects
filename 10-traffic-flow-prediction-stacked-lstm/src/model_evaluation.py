"""Forecasting metrics, residual analysis, and baseline comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def safe_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute MAPE while guarding against division by zero."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = np.clip(np.abs(actual_array), 1e-8, None)
    return float(
        np.mean(np.abs((actual_array - predicted_array) / denominator))
        * 100
    )


def regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate standard held-out forecasting metrics."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mape_pct": safe_mape(actual, predicted),
        "r2": float(r2_score(actual, predicted)),
    }


def comparison_table(
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> pd.DataFrame:
    """Create a recruiter-friendly baseline comparison."""
    return pd.DataFrame(
        [
            {"model": "Persistence baseline", **baseline_metrics},
            {"model": "Stacked LSTM", **model_metrics},
        ]
    )


def residual_frame(
    timestamps: pd.Series,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    """Build a row-level forecast table."""
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "actual_congestion_index": actual_array,
            "predicted_congestion_index": predicted_array,
            "residual": actual_array - predicted_array,
            "absolute_error": np.abs(actual_array - predicted_array),
        }
    )
