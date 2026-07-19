from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud_inference import NumpyBitcoinLSTM
from src.config import CONFIG_PATH, SAMPLE_DATA_PATH, SCALER_PATH, WEIGHTS_PATH
from src.data_preprocessing import clean_market_data
from src.forecasting_pipeline import forecast_future, load_json, load_scaler


REQUIRED_FILES = [
    "README.md",
    "README_HOSTING.md",
    "app/streamlit_app.py",
    "app/requirements.txt",
    "data/bitcoin_price_sample.csv",
    "data/README_data.md",
    "models/bitcoin_lstm_model.keras",
    "models/bitcoin_lstm_weights.npz",
    "models/bitcoin_scaler.pkl",
    "models/best_config.json",
    "models/model_metadata.json",
    "notebooks/bitcoin_price_prediction.ipynb",
    "src/cloud_inference.py",
    "src/data_preprocessing.py",
    "src/feature_engineering.py",
    "src/forecasting_pipeline.py",
    "src/model_evaluation.py",
    "src/model_training.py",
    "src/sequence_generation.py",
    "tests/test_pipeline.py",
    "tests/test_cloud_inference.py",
]


def main() -> None:
    missing = [item for item in REQUIRED_FILES if not (PROJECT_ROOT / item).exists()]
    if missing:
        raise FileNotFoundError(f"Required project files are missing: {missing}")

    sample = clean_market_data(pd.read_csv(SAMPLE_DATA_PATH))
    model = NumpyBitcoinLSTM(WEIGHTS_PATH)
    scaler = load_scaler(SCALER_PATH)
    config = load_json(CONFIG_PATH)
    forecast = forecast_future(
        sample,
        model,
        scaler,
        look_back=int(config["look_back"]),
        horizon=7,
    )

    checks = {
        "required_files": "passed",
        "sample_rows": int(len(sample)),
        "sample_start": str(sample["Date"].min().date()),
        "sample_end": str(sample["Date"].max().date()),
        "forecast_rows": int(len(forecast)),
        "forecast_values_finite": bool(np.isfinite(forecast["Predicted_Close"]).all()),
        "forecast_values_positive": bool((forecast["Predicted_Close"] > 0).all()),
        "cloud_runtime": "NumPy inference; no Keras startup required",
    }

    if not checks["forecast_values_finite"] or not checks["forecast_values_positive"]:
        raise RuntimeError("Forecast validation failed.")

    print(json.dumps(checks, indent=2))
    print("Bitcoin Price Prediction project validation passed.")


if __name__ == "__main__":
    main()
