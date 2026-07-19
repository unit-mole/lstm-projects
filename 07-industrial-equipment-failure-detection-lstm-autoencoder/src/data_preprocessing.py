from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetSchema:
    sensor_cols: list[str]
    unit_id_col: str = "unit_id"
    time_col: str = "cycle"
    label_col: str | None = "failure_label"


def load_sensor_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load CSV sensor data and reject empty inputs."""
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("The sensor dataset is empty.")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def validate_required_columns(frame: pd.DataFrame, schema: DatasetSchema) -> None:
    required = [schema.unit_id_col, schema.time_col, *schema.sensor_cols]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required model columns: " + ", ".join(missing)
        )


def clean_sensor_data(frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Clean a sensor frame without leaking future information across units.

    - Coerces time and sensors to numeric values.
    - Removes rows missing unit/time identifiers.
    - Collapses duplicate unit-time rows by keeping the last record.
    - Interpolates sensor gaps within each unit, then uses forward/backward fill.
    - Preserves chronological order within each unit.
    """
    validate_required_columns(frame, schema)
    clean = frame.copy()
    clean[schema.time_col] = pd.to_numeric(clean[schema.time_col], errors="coerce")
    for column in schema.sensor_cols:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if schema.label_col and schema.label_col in clean.columns:
        clean[schema.label_col] = pd.to_numeric(
            clean[schema.label_col], errors="coerce"
        ).fillna(0).astype(int)

    clean = clean.dropna(subset=[schema.unit_id_col, schema.time_col])
    clean = clean.drop_duplicates(
        subset=[schema.unit_id_col, schema.time_col], keep="last"
    )
    clean = clean.sort_values([schema.unit_id_col, schema.time_col]).reset_index(drop=True)

    clean[schema.sensor_cols] = clean.groupby(schema.unit_id_col)[schema.sensor_cols].transform(
        lambda group: group.interpolate(method="linear", limit_direction="both").ffill().bfill()
    )
    if clean[schema.sensor_cols].isna().any().any():
        missing = clean[schema.sensor_cols].isna().sum()
        raise ValueError(f"Unresolved missing sensor values: {missing[missing > 0].to_dict()}")
    return clean


def split_by_unit(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create leakage-resistant train/validation/test partitions by equipment ID."""
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid split fractions.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    units = sorted(frame[schema.unit_id_col].dropna().unique())
    train_end = int(len(units) * train_fraction)
    validation_end = int(len(units) * (train_fraction + validation_fraction))
    train_units = units[:train_end]
    validation_units = units[train_end:validation_end]
    test_units = units[validation_end:]
    return (
        frame[frame[schema.unit_id_col].isin(train_units)].copy(),
        frame[frame[schema.unit_id_col].isin(validation_units)].copy(),
        frame[frame[schema.unit_id_col].isin(test_units)].copy(),
    )


def fit_training_scaler(frame: pd.DataFrame, sensor_cols: list[str]) -> StandardScaler:
    """Fit a StandardScaler only on the training partition."""
    scaler = StandardScaler()
    scaler.fit(frame[sensor_cols])
    return scaler


def apply_scaler(
    frame: pd.DataFrame,
    sensor_cols: list[str],
    scaler,
) -> pd.DataFrame:
    transformed = frame.copy()
    transformed[sensor_cols] = scaler.transform(frame[sensor_cols])
    return transformed
