from __future__ import annotations

from typing import Any

import numpy as np

from src.inference_pipeline import predict_next_frame


def recursive_forecast(
    model, sequence: np.ndarray, metadata: dict[str, Any], steps: int = 4
) -> np.ndarray:
    """Feed each predicted frame back into the rolling input window."""
    if steps < 1:
        raise ValueError("steps must be at least one")
    rolling = np.asarray(sequence, dtype="float32").copy()
    forecasts = []
    for _ in range(steps):
        predicted = predict_next_frame(model, rolling, metadata)
        forecasts.append(predicted)
        rolling = np.concatenate([rolling[1:], predicted[None, ...]], axis=0)
    return np.asarray(forecasts, dtype="float32")
