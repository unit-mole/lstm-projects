"""Model loading and single-step or recursive multi-step inference."""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

from pathlib import Path

import keras
import numpy as np


def load_prediction_model(model_path: str | Path) -> keras.Model:
    """Load the saved Keras model without recompiling training state."""
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(f"Model artifact not found: {source}")
    return keras.models.load_model(source, compile=False)


def validate_sequence_shape(
    sequence: np.ndarray,
    expected_shape: tuple[int, int, int, int] = (6, 32, 32, 1),
) -> np.ndarray:
    """Validate and normalize a single input sequence."""
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.shape != expected_shape:
        raise ValueError(f"Sequence shape {arr.shape} does not match {expected_shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("Sequence contains NaN or infinite values.")
    return np.clip(arr, 0.0, 1.0)


def predict_next_frame(model: keras.Model, sequence: np.ndarray) -> np.ndarray:
    """Predict one next frame from one ordered input sequence."""
    expected = tuple(int(v) for v in model.input_shape[1:])
    prepared = validate_sequence_shape(sequence, expected)
    prediction = np.asarray(model.predict(prepared[None, ...], verbose=0))[0]
    return np.clip(prediction.astype(np.float32), 0.0, 1.0)


def batch_predict(model: keras.Model, sequences: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Predict a batch of input sequences."""
    arr = np.asarray(sequences, dtype=np.float32)
    expected = tuple(int(v) for v in model.input_shape[1:])
    if arr.ndim != 5 or tuple(arr.shape[1:]) != expected:
        raise ValueError(f"Expected batch shape (n, {expected}); received {arr.shape}")
    return np.clip(np.asarray(model.predict(arr, batch_size=batch_size, verbose=0)), 0.0, 1.0)


def recursive_predict(
    model: keras.Model,
    sequence: np.ndarray,
    future_steps: int = 5,
) -> np.ndarray:
    """Generate future frames by feeding each prediction back into the rolling window."""
    if future_steps < 1:
        raise ValueError("future_steps must be at least 1")
    expected = tuple(int(v) for v in model.input_shape[1:])
    window = validate_sequence_shape(sequence, expected).copy()
    predictions = []
    for _ in range(future_steps):
        next_frame = predict_next_frame(model, window)
        predictions.append(next_frame)
        window = np.concatenate([window[1:], next_frame[None, ...]], axis=0)
    return np.asarray(predictions, dtype=np.float32)
