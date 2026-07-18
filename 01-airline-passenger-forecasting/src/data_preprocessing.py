from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

DATE_ALIASES = (
    "month", "date", "period", "timestamp", "time", "ds", "year_month"
)
TARGET_ALIASES = (
    "passengers", "passenger", "passenger_count", "passenger count",
    "air_passengers", "airline_passengers", "value", "y"
)


def _normalize(name: object) -> str:
    return str(name).strip().lower().replace("-", "_")


def _find_column(columns: list[object], aliases: tuple[str, ...]) -> object | None:
    normalized = {_normalize(col): col for col in columns}
    for alias in aliases:
        key = _normalize(alias)
        if key in normalized:
            return normalized[key]
    return None


def load_tabular_data(source: str | Path | BinaryIO | TextIO | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV source or copy an existing DataFrame."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    if isinstance(source, (BytesIO, StringIO)) or hasattr(source, "read"):
        return pd.read_csv(source)
    raise TypeError("source must be a CSV path, file-like object, or pandas DataFrame")


def prepare_monthly_series(
    data: pd.DataFrame,
    duplicate_strategy: str = "sum",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Standardize a passenger history to one chronologically ordered row per month.

    Duplicate months are aggregated. Missing months are inserted and interpolated
    using time-based interpolation. Notes describing every correction are returned.
    """
    if data.empty:
        raise ValueError("The uploaded dataset is empty.")

    notes: list[str] = []
    date_col = _find_column(list(data.columns), DATE_ALIASES)
    target_col = _find_column(list(data.columns), TARGET_ALIASES)

    if date_col is None and len(data.columns) >= 1:
        date_col = data.columns[0]
        notes.append(f"Used first column '{date_col}' as the month/date column.")
    if target_col is None and len(data.columns) >= 2:
        candidates = [c for c in data.columns if c != date_col]
        target_col = candidates[0]
        notes.append(f"Used column '{target_col}' as the passenger-count column.")

    if date_col is None or target_col is None:
        raise ValueError("A date/month column and passenger-count column are required.")

    frame = data[[date_col, target_col]].copy()
    frame.columns = ["Month", "Passengers"]
    frame["Month"] = pd.to_datetime(frame["Month"], errors="coerce")
    frame["Passengers"] = pd.to_numeric(frame["Passengers"], errors="coerce")

    invalid_rows = int(frame[["Month", "Passengers"]].isna().any(axis=1).sum())
    if invalid_rows:
        frame = frame.dropna(subset=["Month", "Passengers"])
        notes.append(f"Removed {invalid_rows} row(s) with invalid dates or passenger values.")

    if frame.empty:
        raise ValueError("No valid rows remained after parsing the dataset.")
    if (frame["Passengers"] < 0).any():
        raise ValueError("Passenger counts cannot be negative.")

    frame["Month"] = frame["Month"].dt.to_period("M").dt.to_timestamp()
    duplicate_count = int(frame.duplicated("Month", keep=False).sum())
    if duplicate_count:
        if duplicate_strategy not in {"sum", "mean"}:
            raise ValueError("duplicate_strategy must be either 'sum' or 'mean'.")
        frame = (
            frame.groupby("Month", as_index=False)["Passengers"]
            .agg(duplicate_strategy)
        )
        notes.append(
            f"Aggregated {duplicate_count} duplicate monthly row(s) using {duplicate_strategy}."
        )

    frame = frame.sort_values("Month").reset_index(drop=True)
    complete_months = pd.date_range(frame["Month"].min(), frame["Month"].max(), freq="MS")
    missing_count = int(len(complete_months) - len(frame))
    if missing_count:
        frame = frame.set_index("Month").reindex(complete_months)
        frame.index.name = "Month"
        frame["Passengers"] = frame["Passengers"].interpolate(
            method="time", limit_direction="both"
        )
        frame = frame.reset_index()
        notes.append(f"Inserted and interpolated {missing_count} missing month(s).")

    frame["Passengers"] = frame["Passengers"].astype(float)
    frame["Year"] = frame["Month"].dt.year
    frame["MonthNumber"] = frame["Month"].dt.month
    frame["MonthName"] = frame["Month"].dt.strftime("%b")
    frame["TimeIndex"] = np.arange(len(frame), dtype=int)
    return frame, notes


def load_and_prepare(
    source: str | Path | BinaryIO | TextIO | pd.DataFrame,
    duplicate_strategy: str = "sum",
) -> tuple[pd.DataFrame, list[str]]:
    return prepare_monthly_series(load_tabular_data(source), duplicate_strategy)


def chronological_split_points(
    n_rows: int,
    train_ratio: float = 2 / 3,
    validation_ratio: float = 1 / 6,
    minimum_segment: int = 12,
) -> tuple[int, int]:
    """Return exclusive train and validation endpoints for a time-ordered split."""
    if n_rows < minimum_segment * 3:
        raise ValueError(
            f"At least {minimum_segment * 3} monthly observations are required for "
            "train/validation/test evaluation."
        )
    train_end = max(minimum_segment, int(round(n_rows * train_ratio)))
    validation_end = max(train_end + minimum_segment, int(round(n_rows * (train_ratio + validation_ratio))))
    validation_end = min(validation_end, n_rows - minimum_segment)
    return train_end, validation_end
