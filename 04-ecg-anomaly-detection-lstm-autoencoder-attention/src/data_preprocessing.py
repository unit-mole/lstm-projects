from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SEQUENCE_LENGTH

LABEL_ALIASES = ("label", "target", "is_anomaly", "anomaly", "class")
ID_ALIASES = ("signal_id", "record_id", "id", "sample_id")
TYPE_ALIASES = ("anomaly_type", "signal_type", "type")


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _signal_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        column
        for aliases in (LABEL_ALIASES, ID_ALIASES, TYPE_ALIASES)
        for column in [_find_column(list(frame.columns), aliases)]
        if column is not None
    }

    preferred = [
        column
        for column in frame.columns
        if column not in excluded
        and str(column).lower().startswith(
            ("sample_", "signal_", "value_", "timestep_")
        )
    ]
    if len(preferred) >= SEQUENCE_LENGTH:
        return preferred[:SEQUENCE_LENGTH]
    numeric_candidates = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return numeric_candidates[:SEQUENCE_LENGTH]


def prepare_ecg_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize a wide ECG CSV into one row per 140-point sequence."""
    if frame is None or frame.empty:
        raise ValueError("The ECG dataset is empty.")

    data = frame.copy()
    signal_columns = _signal_columns(data)
    if len(signal_columns) != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected {SEQUENCE_LENGTH} numeric signal columns, found {len(signal_columns)}."
        )

    data[signal_columns] = data[signal_columns].apply(pd.to_numeric, errors="coerce")
    data[signal_columns] = data[signal_columns].interpolate(axis=1, limit_direction="both")
    data = data.dropna(subset=signal_columns).reset_index(drop=True)

    if data.empty:
        raise ValueError("No complete ECG sequences remained after missing-value handling.")

    label_column = _find_column(list(data.columns), LABEL_ALIASES)
    id_column = _find_column(list(data.columns), ID_ALIASES)
    type_column = _find_column(list(data.columns), TYPE_ALIASES)

    standardized = pd.DataFrame(
        data[signal_columns].to_numpy(dtype="float32"),
        columns=[f"sample_{index:03d}" for index in range(SEQUENCE_LENGTH)],
    )
    standardized.insert(
        0,
        "signal_id",
        (
            data[id_column].astype(str).to_numpy()
            if id_column is not None
            else np.asarray([f"ECG_{index:04d}" for index in range(len(data))])
        ),
    )

    if label_column is not None:
        standardized.insert(
            1,
            "label",
            pd.to_numeric(data[label_column], errors="coerce")
            .fillna(0)
            .astype(int)
            .clip(0, 1)
            .to_numpy(),
        )
    else:
        standardized.insert(1, "label", -1)

    standardized.insert(
        2,
        "anomaly_type",
        (
            data[type_column].astype(str).to_numpy()
            if type_column is not None
            else np.asarray(["unknown"] * len(data))
        ),
    )

    standardized = standardized.drop_duplicates(
        subset=[f"sample_{index:03d}" for index in range(SEQUENCE_LENGTH)]
    ).reset_index(drop=True)
    return standardized


def load_ecg_csv(path: str | Path) -> pd.DataFrame:
    return prepare_ecg_frame(pd.read_csv(path))


def frame_to_sequences(frame: pd.DataFrame) -> np.ndarray:
    signal_columns = [f"sample_{index:03d}" for index in range(SEQUENCE_LENGTH)]
    return frame[signal_columns].to_numpy(dtype="float32")[..., None]
