"""High-level inference orchestration used by the Streamlit interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_evaluation import calculate_frame_metrics
from .prediction_pipeline import predict_next_frame, recursive_predict


@dataclass(frozen=True)
class InferenceResult:
    input_sequence: np.ndarray
    predicted_next_frame: np.ndarray
    predicted_future_frames: np.ndarray
    actual_next_frame: np.ndarray | None
    absolute_error: np.ndarray | None
    metrics: dict[str, float] | None


def run_inference(
    model,
    sequence: np.ndarray,
    *,
    actual_next_frame: np.ndarray | None = None,
    future_steps: int = 1,
) -> InferenceResult:
    """Run prediction and optional evaluation against an available target frame."""
    predicted_next = predict_next_frame(model, sequence)
    future = (
        predicted_next[None, ...]
        if future_steps == 1
        else recursive_predict(model, sequence, future_steps)
    )
    error = None
    metrics = None
    if actual_next_frame is not None:
        actual = np.asarray(actual_next_frame, dtype=np.float32)
        error = np.abs(actual - predicted_next)
        metrics = calculate_frame_metrics(actual[None, ...], predicted_next[None, ...])
    return InferenceResult(
        input_sequence=np.asarray(sequence, dtype=np.float32),
        predicted_next_frame=predicted_next,
        predicted_future_frames=future,
        actual_next_frame=actual_next_frame,
        absolute_error=error,
        metrics=metrics,
    )
