from __future__ import annotations

import numpy as np
import pandas as pd


def generate_predictive_maintenance_data(
    n_units: int = 120,
    cycles: int = 70,
    n_sensors: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Generate the deterministic turbofan-style dataset used by the original notebook.

    The label becomes 1 during the final five cycles before each simulated failure point.
    Sensor signals include unit-specific baselines, periodic behavior, degradation drift,
    and random measurement noise.
    """
    if n_units < 1 or cycles < 2 or n_sensors < 1:
        raise ValueError("n_units, cycles, and n_sensors must be positive.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    sensor_cols = [f"sensor_{i}" for i in range(1, n_sensors + 1)]

    for unit in range(1, n_units + 1):
        failure_point = rng.integers(int(cycles * 0.7), cycles + 1)
        operating_setting = rng.uniform(0.8, 1.2)
        baseline = rng.normal(0, 0.4, n_sensors)

        for cycle in range(1, cycles + 1):
            health_ratio = max(0.0, 1.0 - cycle / failure_point)
            degradation = 1.0 - health_ratio
            values = []
            for sensor_index in range(n_sensors):
                periodic = baseline[sensor_index] + operating_setting * 0.1 * np.sin(
                    cycle / 6 + sensor_index
                )
                direction = 1 if sensor_index % 2 == 0 else -1
                drift = degradation * rng.uniform(0.5, 1.2) * direction
                noise = rng.normal(0, 0.08)
                values.append(periodic + drift + noise)

            row: dict[str, float | int] = {
                "unit_id": unit,
                "cycle": cycle,
                "failure_label": int(cycle >= failure_point - 5),
            }
            row.update(dict(zip(sensor_cols, values)))
            rows.append(row)

    return pd.DataFrame(rows), sensor_cols
