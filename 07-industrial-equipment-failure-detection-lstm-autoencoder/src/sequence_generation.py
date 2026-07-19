from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_preprocessing import DatasetSchema


@dataclass(frozen=True)
class SequenceBatch:
    sequences: np.ndarray
    labels: np.ndarray | None
    unit_ids: np.ndarray
    end_times: np.ndarray
    start_positions: np.ndarray
    end_positions: np.ndarray


def build_sequences(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    window_size: int,
    step_size: int = 1,
) -> SequenceBatch:
    """Build fixed-length multivariate windows without crossing equipment IDs."""
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if step_size < 1:
        raise ValueError("step_size must be at least 1.")

    sequences, labels, units, end_times, starts, ends = [], [], [], [], [], []
    label_available = bool(schema.label_col and schema.label_col in frame.columns)

    for unit, group in frame.groupby(schema.unit_id_col, sort=True):
        group = group.sort_values(schema.time_col).reset_index(drop=True)
        values = group[schema.sensor_cols].to_numpy(dtype=np.float32)
        group_labels = (
            group[schema.label_col].to_numpy(dtype=int) if label_available else None
        )
        times = group[schema.time_col].to_numpy()
        for end in range(window_size, len(group) + 1, step_size):
            start = end - window_size
            sequences.append(values[start:end])
            if label_available:
                labels.append(group_labels[end - 1])
            units.append(unit)
            end_times.append(times[end - 1])
            starts.append(start)
            ends.append(end - 1)

    if not sequences:
        raise ValueError(
            f"No sequences could be built. Each selected equipment ID needs at least "
            f"{window_size} chronologically ordered rows."
        )

    return SequenceBatch(
        sequences=np.asarray(sequences, dtype=np.float32),
        labels=np.asarray(labels, dtype=int) if label_available else None,
        unit_ids=np.asarray(units),
        end_times=np.asarray(end_times),
        start_positions=np.asarray(starts, dtype=int),
        end_positions=np.asarray(ends, dtype=int),
    )
