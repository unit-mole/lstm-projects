"""Project paths and default artifact locations."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

DEFAULT_SAMPLE_DATA = DATA_DIR / "sample_traffic_flow_data.csv"
DEFAULT_MODEL_PATH = MODEL_DIR / "stacked_lstm_traffic.keras"
DEFAULT_SCALER_PATH = MODEL_DIR / "scalers.json"
DEFAULT_METADATA_PATH = MODEL_DIR / "model_metadata.json"
