import numpy as np
import pandas as pd

from src.data_preprocessing import prepare_traffic_data


def test_preparation_sorts_deduplicates_and_imputes():
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 02:00",
                "2024-01-01 01:00",
                "2024-01-01 01:00",
                "2024-01-01 03:00",
            ],
            "vehicle_count": [130.0, 100.0, 110.0, np.nan],
            "avg_speed": [60.0, 65.0, 64.0, 58.0],
            "occupancy": [0.4, 0.3, 0.31, 0.5],
            "weather_severity": [0.2, 0.1, 0.15, 0.3],
            "congestion_index": [40.0, 30.0, 31.0, 50.0],
        }
    )

    prepared = prepare_traffic_data(frame)

    assert len(prepared) == 3
    assert prepared["timestamp"].is_monotonic_increasing
    assert prepared["vehicle_count"].isna().sum() == 0
    assert {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}.issubset(
        prepared.columns
    )
