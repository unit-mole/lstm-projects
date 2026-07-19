from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_energy_dataset(n_hours: int = 24 * 365 * 2, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2022-01-01", periods=n_hours, freq="h")
    hour = timestamps.hour
    dayofweek = timestamps.dayofweek
    dayofyear = timestamps.dayofyear
    temperature = 18 + 10 * np.sin(2 * np.pi * dayofyear / 365) + 4 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.5, n_hours)
    humidity = 55 + 15 * np.cos(2 * np.pi * dayofyear / 365) - 6 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 3.0, n_hours)
    weekend = (dayofweek >= 5).astype(int)
    business_activity = np.where((hour >= 8) & (hour <= 18), 1.0, 0.35)
    seasonality = 120 + 22 * np.sin(2 * np.pi * dayofyear / 365) + 14 * np.sin(2 * np.pi * hour / 24)
    load = seasonality + 1.9 * temperature - 0.6 * humidity + 28 * business_activity - 10 * weekend + rng.normal(0, 5, n_hours)
    return pd.DataFrame({
        "timestamp": timestamps,
        "energy_load": load,
        "temperature": temperature,
        "humidity": humidity,
    })


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "data" / "hourly_energy_regenerated.csv"
    generate_energy_dataset().to_csv(output, index=False)
    print(f"Saved reproducible synthetic data to {output}")
