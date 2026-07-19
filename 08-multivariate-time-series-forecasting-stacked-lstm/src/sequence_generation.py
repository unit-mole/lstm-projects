from __future__ import annotations

import numpy as np


def build_supervised_sequences(
    features: np.ndarray,
    target: np.ndarray,
    sequence_length: int,
    forecast_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ordered arrays into [samples, time, features] LSTM tensors.

    For each sample, X contains the previous ``sequence_length`` rows and y
    contains the next ``forecast_horizon`` target values. No future row is used
    inside the input window.
    """
    features = np.asarray(features)
    target = np.asarray(target).reshape(-1)
    if len(features) != len(target):
        raise ValueError("Features and target must have the same number of rows.")
    if sequence_length < 1 or forecast_horizon < 1:
        raise ValueError("sequence_length and forecast_horizon must be positive.")
    required_rows = sequence_length + forecast_horizon
    if len(features) < required_rows:
        raise ValueError(f"At least {required_rows} rows are required.")

    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    last_start = len(features) - sequence_length - forecast_horizon + 1
    for start in range(last_start):
        window_end = start + sequence_length
        target_end = window_end + forecast_horizon
        x_values.append(features[start:window_end])
        y_values.append(target[window_end:target_end])

    x = np.asarray(x_values, dtype=np.float32)
    y = np.asarray(y_values, dtype=np.float32)
    if forecast_horizon == 1:
        y = y.reshape(-1)
    return x, y
