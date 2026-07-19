"""Deterministic synthetic traffic data used by the portfolio demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_traffic_data(
    n_steps: int = 24 * 365,
    seed: int = 42,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """Generate hourly traffic signals with commute and weekend patterns."""
    if n_steps <= 24:
        raise ValueError("n_steps must be greater than 24.")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n_steps, freq="h")
    hour = timestamps.hour
    dayofweek = timestamps.dayofweek

    morning_peak = np.exp(-((hour - 8) ** 2) / 6)
    evening_peak = np.exp(-((hour - 17) ** 2) / 8)
    weekend = (dayofweek >= 5).astype(int)

    weather_severity = np.clip(rng.normal(0.25, 0.18, n_steps), 0, 1)
    vehicle_count = (
        180
        + 210 * morning_peak
        + 240 * evening_peak
        - 60 * weekend
        + rng.normal(0, 15, n_steps)
    )
    average_speed = (
        72
        - 18 * morning_peak
        - 22 * evening_peak
        - 8 * weather_severity
        + rng.normal(0, 2.5, n_steps)
    )
    occupancy = np.clip(
        0.25
        + 0.35 * morning_peak
        + 0.42 * evening_peak
        + 0.10 * weather_severity
        + rng.normal(0, 0.04, n_steps),
        0,
        1,
    )
    congestion_index = (
        0.45 * (vehicle_count / vehicle_count.max())
        + 0.35 * occupancy
        + 0.20 * (1 - average_speed / max(average_speed.max(), 1))
    ) * 100 + rng.normal(0, 2.5, n_steps)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "vehicle_count": vehicle_count,
            "avg_speed": average_speed,
            "occupancy": occupancy,
            "weather_severity": weather_severity,
            "congestion_index": congestion_index,
        }
    )
