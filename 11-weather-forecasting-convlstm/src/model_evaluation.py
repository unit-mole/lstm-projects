from __future__ import annotations

import numpy as np


def map_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def map_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(residual**2)))


def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    true_event = np.asarray(y_true) >= threshold
    pred_event = np.asarray(y_pred) >= threshold
    intersection = np.logical_and(true_event, pred_event).sum()
    union = np.logical_or(true_event, pred_event).sum()
    true_positive = intersection
    false_positive = np.logical_and(~true_event, pred_event).sum()
    false_negative = np.logical_and(true_event, ~pred_event).sum()
    iou = float(intersection / union) if union else 1.0
    pixel_accuracy = float(np.mean(true_event == pred_event))
    pod = float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0
    far = float(false_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0
    csi = float(true_positive / (true_positive + false_positive + false_negative)) if true_positive + false_positive + false_negative else 0.0
    return {"iou": iou, "pixel_accuracy": pixel_accuracy, "pod": pod, "far": far, "csi": csi}


def structural_similarity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    true_map = np.squeeze(np.asarray(y_true, dtype="float32"))
    pred_map = np.squeeze(np.asarray(y_pred, dtype="float32"))
    return float(structural_similarity(true_map, pred_map, data_range=1.0))


def evaluate_weather_map(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    results = {"mae": map_mae(y_true, y_pred), "rmse": map_rmse(y_true, y_pred)}
    results.update(threshold_metrics(y_true, y_pred, threshold))
    if np.asarray(y_true).ndim <= 4:
        results["ssim"] = structural_similarity_score(y_true, y_pred)
    return results
