"""CI smoke test for the model, scaler, and metadata artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SAMPLE_DATA  # noqa: E402
from src.config import MODEL_DIR  # noqa: E402
from src.data_preprocessing import prepare_traffic_data  # noqa: E402
from src.forecasting_pipeline import TrafficForecastingPipeline  # noqa: E402


def main() -> None:
    pipeline = TrafficForecastingPipeline.from_artifacts(MODEL_DIR)
    sample = prepare_traffic_data(pd.read_csv(DEFAULT_SAMPLE_DATA))
    prediction = pipeline.predict_next(sample)
    if not np.isfinite(prediction):
        raise RuntimeError("The model produced a non-finite prediction.")
    print(f"Artifact validation passed. Prediction: {prediction:.4f}")


if __name__ == "__main__":
    main()
