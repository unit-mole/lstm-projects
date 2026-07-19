from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = MODEL_DIR / "convlstm_weather_forecast.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SAMPLE_DATA_PATH = DATA_DIR / "sample_weather_sequences.npz"
DEFAULT_INPUT_FRAMES = 6
DEFAULT_GRID_SIZE = 24
DEFAULT_CHANNELS = 1
DEFAULT_SEED = 42
