from __future__ import annotations

import numpy as np
import pandas as pd

from .sequence_generation import SequenceBatch
from .thresholding import health_status, risk_interpretation


def sequence_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    metric: str = "mae",
) -> np.ndarray:
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed sequences must have the same shape.")
    difference = original - reconstructed
    if metric == "mae":
        return np.mean(np.abs(difference), axis=(1, 2))
    if metric == "mse":
        return np.mean(np.square(difference), axis=(1, 2))
    raise ValueError("metric must be 'mae' or 'mse'.")


def sensor_reconstruction_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    return np.mean(np.abs(original - reconstructed), axis=1)


def build_prediction_frame(
    batch: SequenceBatch,
    errors: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    result = pd.DataFrame({
        "sequence_index": np.arange(len(errors)),
        "unit_id": batch.unit_ids,
        "window_end": batch.end_times,
        "reconstruction_error": errors,
        "anomaly_threshold": threshold,
        "anomaly_score": errors / threshold,
        "predicted_anomaly": (errors > threshold).astype(int),
    })
    if batch.labels is not None:
        result["true_label"] = batch.labels
    result["health_status"] = [health_status(float(e), threshold) for e in errors]
    result["risk_interpretation"] = [
        risk_interpretation(float(e), threshold) for e in errors
    ]
    return result
