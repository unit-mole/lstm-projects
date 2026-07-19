from __future__ import annotations

import numpy as np


def create_rolling_sequences(
    frames: np.ndarray, input_frames: int = 6, forecast_horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one chronologically ordered frame stream into supervised sequences."""
    values = np.asarray(frames, dtype="float32")
    if values.ndim == 3:
        values = values[..., None]
    if values.ndim != 4:
        raise ValueError("frames must have shape [time, height, width, channels]")
    if input_frames < 1 or forecast_horizon < 1:
        raise ValueError("input_frames and forecast_horizon must be positive")
    last_start = len(values) - input_frames - forecast_horizon + 1
    if last_start < 1:
        raise ValueError("not enough frames to form a sequence")
    X, y = [], []
    for start in range(last_start):
        target_index = start + input_frames + forecast_horizon - 1
        X.append(values[start:start + input_frames])
        y.append(values[target_index])
    return np.asarray(X), np.asarray(y)
