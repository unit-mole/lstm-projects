from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    if len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted arrays must have equal length.")

    nonzero = np.abs(actual_values) > 1e-12
    mape = (
        float(np.mean(np.abs((actual_values[nonzero] - predicted_values[nonzero]) / actual_values[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "MAE": float(mean_absolute_error(actual_values, predicted_values)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "MAPE": mape,
        "R2": float(r2_score(actual_values, predicted_values)),
    }


def baseline_predictions(close_series: pd.Series) -> pd.DataFrame:
    """Transparent one-step baselines aligned to the supplied close series."""
    close = pd.Series(close_series, dtype=float).reset_index(drop=True)
    result = pd.DataFrame({"Actual": close})
    result["Naive"] = close.shift(1)
    result["Moving_Average_7"] = close.shift(1).rolling(7).mean()

    linear = np.full(len(close), np.nan, dtype=float)
    window = 30
    for index in range(window, len(close)):
        y = close.iloc[index - window : index].to_numpy()
        x = np.arange(window, dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        linear[index] = intercept + slope * window
    result["Linear_Trend_30"] = linear
    return result
