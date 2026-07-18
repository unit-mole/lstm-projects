from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

SAMPLE_DATA_PATH = DATA_DIR / "airline_passengers_sample.csv"
MODEL_PATH = MODEL_DIR / "airline_passenger_lstm.keras"
SCALER_PATH = MODEL_DIR / "seasonal_growth_scaler.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
BEST_CONFIG_PATH = MODEL_DIR / "best_config.json"

DATE_COLUMN = "Month"
TARGET_COLUMN = "Passengers"
SEASONAL_PERIOD = 12
LOOKBACK = 12
EFFECTIVE_RAW_HISTORY = SEASONAL_PERIOD + LOOKBACK
RANDOM_SEED = 42
