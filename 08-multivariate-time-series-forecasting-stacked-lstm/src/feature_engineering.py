from __future__ import annotations

import numpy as np
import pandas as pd


ENGINEERED_FEATURES = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "weekend")


def add_calendar_features(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Create causal calendar features from each row's timestamp."""
    if timestamp_column not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="raise")
    result["hour"] = result[timestamp_column].dt.hour
    result["dayofweek"] = result[timestamp_column].dt.dayofweek
    result["month"] = result[timestamp_column].dt.month
    result["weekend"] = (result["dayofweek"] >= 5).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)
    return result


def validate_feature_columns(frame: pd.DataFrame, feature_columns: list[str] | tuple[str, ...]) -> None:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
