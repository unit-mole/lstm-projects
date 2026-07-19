from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "images"

SAMPLE_DATA_PATH = DATA_DIR / "sample_ecg_signals.csv"
WEIGHTS_PATH = MODELS_DIR / "lstm_autoencoder_ecg_weights.npz"
SUPPLIED_MODEL_PATH = MODELS_DIR / "lstm_autoencoder_ecg.keras"
META_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"

SEQUENCE_LENGTH = 140
NUMBER_OF_FEATURES = 1
THRESHOLD = 0.0321530313231051

HEALTHCARE_DISCLAIMER = (
    "Educational and portfolio demonstration only. This application is not a medical "
    "diagnostic tool and must not be used to diagnose heart conditions, make treatment "
    "decisions, or replace qualified clinical interpretation. Predictions may be incorrect."
)

ATTENTION_QUALIFICATION = (
    "The supplied pretrained artifact is a stacked LSTM Autoencoder without a trainable "
    "attention layer. The deployed temporal-focus view is post-hoc explainability derived "
    "from pointwise reconstruction error. Optional retraining code defines a true trainable "
    "temporal-attention architecture."
)
