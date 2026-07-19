"""Reproduce the supplied model's held-out synthetic test metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DIR  # noqa: E402
from src.forecasting_pipeline import TrafficForecastingPipeline  # noqa: E402
from src.synthetic_data import generate_traffic_data  # noqa: E402


def main() -> None:
    frame = generate_traffic_data()
    test_start = int(len(frame) * 0.85)
    test_frame = frame.iloc[test_start:].copy()
    pipeline = TrafficForecastingPipeline.from_artifacts(MODEL_DIR)
    result = pipeline.backtest(test_frame)
    print(json.dumps(result["model_metrics"], indent=2))


if __name__ == "__main__":
    main()
