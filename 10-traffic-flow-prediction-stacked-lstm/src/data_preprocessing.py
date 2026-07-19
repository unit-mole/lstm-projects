"""Traffic CSV loading, schema validation, cleaning, and time ordering."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd

from .feature_engineering import BASE_NUMERIC_COLUMNS
from .feature_engineering import add_time_features
from .feature_engineering import validate_required_columns

DataSource = str | Path | BinaryIO | TextIO


def load_traffic_csv(source: DataSource) -> pd.DataFrame:
    """Load a traffic CSV from a path or file-like object."""
    return pd.read_csv(source)


def infer_timestamp_column(frame: pd.DataFrame) -> str:
    """Identify a timestamp-like column without silently choosing a target."""
    preferred = ["timestamp", "datetime", "date_time", "date", "time"]
    lower_to_original = {column.lower(): column for column in frame.columns}
    for candidate in preferred:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    timestamp_like = [
        column
        for column in frame.columns
        if "time" in column.lower() or "date" in column.lower()
    ]
    if len(timestamp_like) == 1:
        return timestamp_like[0]
    raise ValueError(
        "A timestamp column could not be identified. "
        "Rename it to 'timestamp' or select it in the application."
    )


def prepare_traffic_data(
    frame: pd.DataFrame,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    """Clean traffic data while preserving chronological order."""
    if frame.empty:
        raise ValueError("The traffic dataset is empty.")

    result = frame.copy()
    timestamp_column = timestamp_column or infer_timestamp_column(result)
    if timestamp_column != "timestamp":
        result = result.rename(columns={timestamp_column: "timestamp"})

    validate_required_columns(result, BASE_NUMERIC_COLUMNS)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result = result.dropna(subset=["timestamp"])
    result = result.drop_duplicates(subset=["timestamp"], keep="last")
    result = result.sort_values("timestamp").reset_index(drop=True)

    for column in BASE_NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    numeric_columns = list(BASE_NUMERIC_COLUMNS)
    result[numeric_columns] = result[numeric_columns].interpolate(
        method="linear",
        limit_direction="both",
    )
    result[numeric_columns] = result[numeric_columns].ffill().bfill()

    if result[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing numeric values remain after interpolation. "
            "Review the uploaded traffic data."
        )

    result["occupancy"] = result["occupancy"].clip(0, 1)
    result["weather_severity"] = result["weather_severity"].clip(0, 1)
    return add_time_features(result)


def missing_value_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages."""
    summary = frame.isna().sum().rename("missing_count").to_frame()
    summary["missing_pct"] = (
        summary["missing_count"] / max(len(frame), 1) * 100
    ).round(3)
    return summary.sort_values("missing_count", ascending=False)
