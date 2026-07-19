from __future__ import annotations

import numpy as np


def repair_weather_array(array: np.ndarray) -> np.ndarray:
    """Convert to float32, replace non-finite values, and clip intensities to [0, 1]."""
    values = np.asarray(array, dtype="float32").copy()
    if values.size == 0:
        raise ValueError("weather array cannot be empty")
    if np.isfinite(values).any():
        fill_value = float(np.nanmedian(values[np.isfinite(values)]))
    else:
        fill_value = 0.0
    values[~np.isfinite(values)] = fill_value
    return np.clip(values, 0.0, 1.0)


def validate_sequence_shape(
    sequence: np.ndarray,
    input_frames: int = 6,
    height: int = 24,
    width: int = 24,
    channels: int = 1,
) -> np.ndarray:
    values = repair_weather_array(sequence)
    expected = (input_frames, height, width, channels)
    if values.shape != expected:
        raise ValueError(f"Expected one sequence with shape {expected}, received {values.shape}")
    return values
