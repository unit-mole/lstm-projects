"""Leakage-safe time and traffic feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_NUMERIC_COLUMNS = [
    "vehicle_count",
    "avg_speed",
    "occupancy",
    "weather_severity",
    "congestion_index",
]


def add_time_features(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Add cyclical time features using each row's timestamp only."""
    result = frame.copy()
    timestamps = pd.to_datetime(result[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("The timestamp column contains invalid values.")

    result["hour"] = timestamps.dt.hour
    result["dayofweek"] = timestamps.dt.dayofweek
    result["weekend"] = (result["dayofweek"] >= 5).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)
    return result


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Raise a readable error when required model inputs are absent."""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
