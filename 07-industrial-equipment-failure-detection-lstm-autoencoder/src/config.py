from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def model_dir(self) -> Path:
        return self.root / "models"

    @property
    def output_dir(self) -> Path:
        return self.root / "outputs"


PATHS = ProjectPaths()
DEFAULT_SAMPLE_DATA = PATHS.data_dir / "sample_equipment_sensor_data.csv"
DEFAULT_MODEL = PATHS.model_dir / "lstm_autoencoder_predictive_maintenance.keras"
DEFAULT_METADATA = PATHS.model_dir / "model_metadata.json"
DEFAULT_SCALER = PATHS.model_dir / "scaler.pkl"
DEFAULT_PORTABLE_SCALER = PATHS.model_dir / "scaler.json"
