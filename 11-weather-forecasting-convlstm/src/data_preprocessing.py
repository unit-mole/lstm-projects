from __future__ import annotations

from pathlib import Path

import numpy as np

from src.weather_preprocessing import repair_weather_array


def load_weather_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    with np.load(Path(path), allow_pickle=False) as payload:
        if "X" not in payload:
            raise ValueError("NPZ file must contain an 'X' array")
        X = repair_weather_array(payload["X"])
        y = repair_weather_array(payload["y"]) if "y" in payload else None
        future_y = repair_weather_array(payload["future_y"]) if "future_y" in payload else None
    if X.ndim != 5:
        raise ValueError("X must have shape [samples, time, height, width, channels]")
    return X, y, future_y


def split_independent_sequences(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, ...]:
    """Use a deterministic contiguous split; never shuffle an externally ordered dataset."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    train_end = int(len(X) * train_fraction)
    validation_end = train_end + int(len(X) * validation_fraction)
    if train_end < 1 or validation_end >= len(X):
        raise ValueError("dataset is too small for the requested split")
    return (
        X[:train_end], y[:train_end],
        X[train_end:validation_end], y[train_end:validation_end],
        X[validation_end:], y[validation_end:],
    )
