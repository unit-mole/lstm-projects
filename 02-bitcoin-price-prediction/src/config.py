from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SAMPLE_DATA_PATH = DATA_DIR / "bitcoin_price_sample.csv"
MODEL_PATH = MODELS_DIR / "bitcoin_lstm_model.keras"
WEIGHTS_PATH = MODELS_DIR / "bitcoin_lstm_weights.npz"
SCALER_PATH = MODELS_DIR / "bitcoin_scaler.pkl"
CONFIG_PATH = MODELS_DIR / "best_config.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"

DATE_COLUMN = "Date"
TARGET_COLUMN = "Close"
FEATURE_COLUMNS = ["Close", "SMA_7", "SMA_30", "Return", "Volume"]
DEFAULT_LOOK_BACK = 30
DEFAULT_FORECAST_HORIZON = 7
SUPPORTED_HORIZONS = [1, 7, 14, 30]

FINANCIAL_DISCLAIMER = (
    "This project is for educational and portfolio demonstration purposes only. "
    "It is not financial advice. Bitcoin and cryptocurrency prices are highly "
    "volatile and difficult to predict. Model outputs should not be used for "
    "investment, trading, or financial decisions."
)
