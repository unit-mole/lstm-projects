from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attention_layer import temporal_focus_weights


@dataclass
class DetectionResult:
    reconstruction: np.ndarray
    pointwise_error: np.ndarray
    reconstruction_error: float
    threshold: float
    anomaly_score: float
    predicted_label: int
    predicted_status: str
    temporal_focus: np.ndarray


def reconstruction_errors(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    values = np.asarray(original, dtype=float)
    estimates = np.asarray(reconstructed, dtype=float)
    return np.mean(np.abs(values - estimates), axis=(1, 2))


def classify_reconstruction(
    original_signal: np.ndarray,
    reconstructed_signal: np.ndarray,
    threshold: float,
) -> DetectionResult:
    original = np.asarray(original_signal, dtype=float).reshape(-1)
    reconstructed = np.asarray(reconstructed_signal, dtype=float).reshape(-1)
    pointwise = np.abs(original - reconstructed)
    error = float(pointwise.mean())
    predicted_label = int(error >= threshold)
    return DetectionResult(
        reconstruction=reconstructed,
        pointwise_error=pointwise,
        reconstruction_error=error,
        threshold=float(threshold),
        anomaly_score=float(error / threshold),
        predicted_label=predicted_label,
        predicted_status="Anomalous pattern" if predicted_label else "Normal pattern",
        temporal_focus=temporal_focus_weights(original, reconstructed),
    )
