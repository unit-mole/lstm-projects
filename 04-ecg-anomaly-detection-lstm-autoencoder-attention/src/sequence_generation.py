from __future__ import annotations

import numpy as np

from .config import NUMBER_OF_FEATURES, SEQUENCE_LENGTH


def ensure_model_shape(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype="float32")

    if values.ndim == 1:
        values = values[None, :, None]
    elif values.ndim == 2:
        if values.shape == (SEQUENCE_LENGTH, NUMBER_OF_FEATURES):
            values = values[None, :, :]
        elif values.shape[1] == SEQUENCE_LENGTH:
            values = values[..., None]
        else:
            raise ValueError(
                "Two-dimensional ECG input must be one sequence shaped "
                f"({SEQUENCE_LENGTH}, {NUMBER_OF_FEATURES}) or a batch shaped "
                f"(rows, {SEQUENCE_LENGTH})."
            )

    if values.ndim != 3:
        raise ValueError("ECG input must have shape (rows, timesteps, features).")

    if values.shape[1:] != (SEQUENCE_LENGTH, NUMBER_OF_FEATURES):
        raise ValueError(
            f"Expected shape (*, {SEQUENCE_LENGTH}, {NUMBER_OF_FEATURES}), "
            f"received {values.shape}."
        )
    return values
