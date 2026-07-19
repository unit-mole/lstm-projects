from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold(
    train_errors: np.ndarray,
    method: str = "mean_plus_3std",
    percentile: float = 99.0,
) -> float:
    values = np.asarray(train_errors, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("train_errors must contain finite values.")
    if method == "mean_plus_3std":
        return float(values.mean() + 3.0 * values.std())
    if method == "percentile":
        if not 0 < percentile < 100:
            raise ValueError("percentile must be between 0 and 100.")
        return float(np.percentile(values, percentile))
    raise ValueError(f"Unsupported threshold method: {method}")


def optimize_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Supervised diagnostic option; never use labels as model input."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("At least two classes are required to optimize a threshold.")
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def health_status(error: float, threshold: float) -> str:
    if error <= threshold:
        return "Normal Operation"
    if error <= 1.5 * threshold:
        return "Warning / Elevated Anomaly Score"
    return "Potential Failure / High-Risk Anomaly"


def risk_interpretation(error: float, threshold: float) -> str:
    status = health_status(error, threshold)
    if status == "Normal Operation":
        return (
            "The sequence is reconstructed with relatively low error and resembles "
            "the normal operating patterns learned by the model."
        )
    if status.startswith("Warning"):
        return (
            "The sequence is above the learned anomaly threshold. Review sensor context, "
            "recent maintenance, and operating conditions before taking action."
        )
    return (
        "The reconstruction error is substantially above the normal-behavior threshold. "
        "This is a high-risk anomaly signal requiring qualified human investigation."
    )
