from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape.")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mape_pct": float(mean_absolute_percentage_error(actual, predicted) * 100),
        "r2": float(r2_score(actual, predicted)),
    }


def naive_previous_value(target_values, sequence_length: int = 24) -> np.ndarray:
    target_values = np.asarray(target_values).reshape(-1)
    if len(target_values) <= sequence_length:
        raise ValueError("Not enough observations for the requested sequence length.")
    return target_values[sequence_length - 1:-1]


def residuals(actual, predicted) -> np.ndarray:
    return np.asarray(actual).reshape(-1) - np.asarray(predicted).reshape(-1)
