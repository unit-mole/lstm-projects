"""Pixel, structural, and foreground-aware evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calculate_frame_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate image-quality metrics across a batch of next-frame predictions."""
    truth = np.asarray(y_true, dtype=np.float32)
    prediction = np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: true={truth.shape}, predicted={prediction.shape}")
    if truth.ndim != 4:
        raise ValueError("Expected arrays shaped (samples, height, width, channels)")

    mse = float(np.mean((truth - prediction) ** 2))
    mae = float(np.mean(np.abs(truth - prediction)))
    rmse = float(np.sqrt(mse))
    ssim = float(np.mean([
        structural_similarity(truth[i, ..., 0], prediction[i, ..., 0], data_range=1.0)
        for i in range(len(truth))
    ]))
    psnr_values = []
    for i in range(len(truth)):
        sample_mse = float(np.mean((truth[i] - prediction[i]) ** 2))
        psnr_values.append(
            float("inf")
            if sample_mse == 0.0
            else float(peak_signal_noise_ratio(
                truth[i, ..., 0], prediction[i, ..., 0], data_range=1.0
            ))
        )
    psnr = float(np.mean(psnr_values))

    true_binary = (truth >= threshold).astype(np.float32)
    pred_binary = (prediction >= threshold).astype(np.float32)
    intersection = (true_binary * pred_binary).reshape(len(truth), -1).sum(axis=1)
    union = (
        true_binary.reshape(len(truth), -1).sum(axis=1)
        + pred_binary.reshape(len(truth), -1).sum(axis=1)
        - intersection
    )
    iou = float(np.mean((intersection + 1e-6) / (union + 1e-6)))
    pixel_accuracy = float(np.mean(true_binary == pred_binary))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "ssim": ssim,
        "psnr_db": psnr,
        "iou": iou,
        "pixel_accuracy": pixel_accuracy,
    }


def persistence_baseline(sequence_batch: np.ndarray) -> np.ndarray:
    """Use the last observed frame as the predicted next frame."""
    sequence_batch = np.asarray(sequence_batch, dtype=np.float32)
    return sequence_batch[:, -1]


def frame_average_baseline(sequence_batch: np.ndarray) -> np.ndarray:
    """Use the average of all input frames as the prediction."""
    sequence_batch = np.asarray(sequence_batch, dtype=np.float32)
    return sequence_batch.mean(axis=1)


def build_comparison_table(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Return a recruiter-friendly baseline comparison table."""
    rows = []
    for model_name, values in predictions.items():
        rows.append({"Model / Approach": model_name, **calculate_frame_metrics(y_true, values)})
    return pd.DataFrame(rows)
