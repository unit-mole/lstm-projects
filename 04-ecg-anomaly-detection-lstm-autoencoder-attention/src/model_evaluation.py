from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    anomaly_scores: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(anomaly_scores, dtype=float)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_anomaly": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall_anomaly": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "f1_anomaly": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
