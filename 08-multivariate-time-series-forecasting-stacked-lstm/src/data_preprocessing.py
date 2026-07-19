from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    rows: int
    columns: int
    start_timestamp: str
    end_timestamp: str
    duplicate_timestamps_removed: int
    missing_values_before: dict[str, int]
    missing_values_after: dict[str, int]


def load_time_series_csv(path_or_buffer: str | Path | object) -> pd.DataFrame:
    """Load a CSV while keeping column inference explicit and testable."""
    return pd.read_csv(path_or_buffer)


def prepare_time_series(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
    required_numeric_columns: Iterable[str] = ("energy_load", "temperature", "humidity"),
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Parse, sort and clean time-series data without using future target values.

    Duplicate timestamps keep the last observation. Numeric gaps are filled using
    forward fill, followed by a training-safe median fallback for leading gaps.
    In production, fit the fallback medians on the training partition and pass them
    explicitly rather than deriving them from the entire dataset.
    """
    if frame.empty:
        raise ValueError("The input dataset is empty.")
    if timestamp_column not in frame.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")

    result = frame.copy()
    result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors="coerce")
    invalid_timestamps = int(result[timestamp_column].isna().sum())
    if invalid_timestamps:
        raise ValueError(f"{invalid_timestamps} timestamp values could not be parsed.")

    required_numeric_columns = tuple(required_numeric_columns)
    missing_columns = [column for column in required_numeric_columns if column not in result.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in required_numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.sort_values(timestamp_column, kind="stable")
    duplicate_count = int(result.duplicated(subset=[timestamp_column]).sum())
    result = result.drop_duplicates(subset=[timestamp_column], keep="last").reset_index(drop=True)

    missing_before = {key: int(value) for key, value in result.isna().sum().items()}
    numeric_columns = list(result.select_dtypes(include=np.number).columns)
    result[numeric_columns] = result[numeric_columns].ffill()
    for column in numeric_columns:
        if result[column].isna().any():
            median = result[column].median()
            if pd.isna(median):
                raise ValueError(f"Column {column!r} contains no usable numeric values.")
            result[column] = result[column].fillna(median)
    missing_after = {key: int(value) for key, value in result.isna().sum().items()}

    report = DataQualityReport(
        rows=len(result),
        columns=len(result.columns),
        start_timestamp=str(result[timestamp_column].min()),
        end_timestamp=str(result[timestamp_column].max()),
        duplicate_timestamps_removed=duplicate_count,
        missing_values_before=missing_before,
        missing_values_after=missing_after,
    )
    return result, report


def chronological_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split in chronological order; time-series data is never shuffled."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must leave a non-empty test partition.")

    train_end = int(len(frame) * train_ratio)
    validation_end = int(len(frame) * (train_ratio + validation_ratio))
    if train_end == 0 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError("Dataset is too small for the requested split ratios.")

    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )
