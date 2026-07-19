from __future__ import annotations

import numpy as np


def create_sequences(
    scaled_data: np.ndarray,
    look_back: int,
    target_col_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create chronological sliding windows without shuffling."""
    if scaled_data.ndim != 2:
        raise ValueError("scaled_data must be a 2D array.")
    if len(scaled_data) <= look_back:
        raise ValueError("The prepared data is shorter than the requested look-back window.")

    X, y = [], []
    for index in range(len(scaled_data) - look_back):
        X.append(scaled_data[index : index + look_back])
        y.append(scaled_data[index + look_back, target_col_index])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def chronological_split_indices(
    n_rows: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[int, int]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test period.")

    train_end = int(n_rows * train_fraction)
    validation_end = int(n_rows * (train_fraction + validation_fraction))
    return train_end, validation_end
