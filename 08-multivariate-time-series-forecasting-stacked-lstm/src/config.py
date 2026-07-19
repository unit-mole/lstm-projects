from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ForecastConfig:
    timestamp_column: str = "timestamp"
    target_column: str = "energy_load"
    feature_columns: tuple[str, ...] = (
        "energy_load",
        "temperature",
        "humidity",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "weekend",
    )
    sequence_length: int = 24
    forecast_horizon: int = 1
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    seed: int = 42


DEFAULT_CONFIG = ForecastConfig()
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
