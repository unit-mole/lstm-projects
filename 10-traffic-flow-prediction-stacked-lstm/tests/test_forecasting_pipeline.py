from pathlib import Path

import pandas as pd

from src.forecasting_pipeline import TrafficForecastingPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_recursive_forecast_has_requested_horizon():
    pipeline = TrafficForecastingPipeline.from_artifacts(
        PROJECT_ROOT / "models"
    )
    frame = pd.read_csv(
        PROJECT_ROOT / "data" / "sample_traffic_flow_data.csv"
    )

    forecast = pipeline.recursive_forecast(frame, horizon=4)

    assert len(forecast) == 4
    assert forecast["timestamp"].is_monotonic_increasing
    assert forecast["predicted_congestion_index"].notna().all()
