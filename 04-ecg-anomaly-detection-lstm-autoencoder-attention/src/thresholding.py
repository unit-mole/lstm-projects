from __future__ import annotations

import numpy as np
from sklearn.metrics import fbeta_score


def mean_plus_std_threshold(
    normal_errors: np.ndarray,
    standard_deviations: float = 3.0,
) -> float:
    values = np.asarray(normal_errors, dtype=float)
    return float(values.mean() + standard_deviations * values.std())


def percentile_threshold(
    normal_errors: np.ndarray,
    percentile: float = 99.5,
) -> float:
    return float(np.percentile(np.asarray(normal_errors, dtype=float), percentile))


def optimize_threshold_for_recall(
    labels: np.ndarray,
    errors: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, float]:
    """Select a validation threshold using F-beta to emphasize recall."""
    labels = np.asarray(labels, dtype=int)
    errors = np.asarray(errors, dtype=float)
    candidates = np.unique(errors)
    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in candidates:
        predictions = (errors >= threshold).astype(int)
        score = fbeta_score(labels, predictions, beta=beta, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score
