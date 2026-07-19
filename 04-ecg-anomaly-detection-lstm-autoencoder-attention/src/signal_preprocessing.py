from __future__ import annotations

import numpy as np


def interpolate_signal(signal: np.ndarray, sequence_length: int = 140) -> np.ndarray:
    """Resample a one-dimensional signal to the model's required length."""
    values = np.asarray(signal, dtype=float).reshape(-1)
    if len(values) < 2:
        raise ValueError("At least two signal values are required.")
    if not np.isfinite(values).all():
        indices = np.arange(len(values))
        valid = np.isfinite(values)
        if valid.sum() < 2:
            raise ValueError("The signal does not contain enough valid numeric values.")
        values = np.interp(indices, indices[valid], values[valid])

    source_axis = np.linspace(0.0, 1.0, len(values))
    target_axis = np.linspace(0.0, 1.0, sequence_length)
    return np.interp(target_axis, source_axis, values).astype("float32")


def validate_signal_amplitude(
    signal: np.ndarray,
    maximum_absolute_amplitude: float = 10.0,
) -> np.ndarray:
    values = np.asarray(signal, dtype="float32").reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("The signal contains non-finite values.")
    if np.max(np.abs(values)) > maximum_absolute_amplitude:
        raise ValueError(
            "Signal amplitude is outside the safe demonstration range. "
            "Confirm the units and preprocessing."
        )
    return values


def pointwise_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    return np.abs(
        np.asarray(original, dtype=float).reshape(-1)
        - np.asarray(reconstructed, dtype=float).reshape(-1)
    )
