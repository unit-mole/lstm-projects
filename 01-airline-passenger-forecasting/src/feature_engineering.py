from __future__ import annotations

import numpy as np
import pandas as pd


def log_passenger_levels(passengers: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(passengers, dtype=np.float32)
    if np.any(values < 0):
        raise ValueError("Passenger counts cannot be negative.")
    return np.log1p(values)


def seasonal_log_difference(
    passengers: np.ndarray | pd.Series,
    seasonal_period: int = 12,
) -> np.ndarray:
    """Compute log(y_t + 1) - log(y_(t-seasonal_period) + 1)."""
    log_values = log_passenger_levels(passengers)
    differences = np.full(log_values.shape, np.nan, dtype=np.float32)
    differences[seasonal_period:] = (
        log_values[seasonal_period:] - log_values[:-seasonal_period]
    )
    return differences


def cyclical_month_features(month_numbers: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    months = np.asarray(month_numbers, dtype=np.float32)
    angle = 2.0 * np.pi * (months - 1.0) / 12.0
    return np.sin(angle), np.cos(angle)


def add_eda_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["LogPassengers"] = log_passenger_levels(result["Passengers"])
    result["Lag1"] = result["Passengers"].shift(1)
    result["Lag12"] = result["Passengers"].shift(12)
    result["RollingMean12"] = result["Passengers"].rolling(12).mean()
    result["RollingStd12"] = result["Passengers"].rolling(12).std()
    result["SeasonalLogGrowth"] = seasonal_log_difference(result["Passengers"], 12)
    month_sin, month_cos = cyclical_month_features(result["MonthNumber"])
    result["MonthSin"] = month_sin
    result["MonthCos"] = month_cos
    return result
