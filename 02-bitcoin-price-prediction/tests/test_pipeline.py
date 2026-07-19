from __future__ import annotations

import numpy as np
import pandas as pd

from src.cloud_inference import NumpyBitcoinLSTM
from src.config import (
    CONFIG_PATH,
    SAMPLE_DATA_PATH,
    SCALER_PATH,
    WEIGHTS_PATH,
)
from src.data_preprocessing import clean_market_data
from src.feature_engineering import FEATURE_COLUMNS, create_market_features
from src.forecasting_pipeline import forecast_future, load_json, load_scaler, replay_predictions
from src.sequence_generation import create_sequences


def test_cleaning_restores_daily_order_and_ohlcv_columns() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-01", "2024-01-01"],
            "Close": [110.0, 100.0, 101.0],
            "Volume": [300.0, 100.0, 120.0],
        }
    )

    cleaned = clean_market_data(raw)

    assert list(cleaned.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert cleaned["Date"].is_monotonic_increasing
    assert len(cleaned) == 3
    assert cleaned.isna().sum().sum() == 0
    assert (cleaned["High"] >= cleaned[["Open", "Close"]].max(axis=1)).all()
    assert (cleaned["Low"] <= cleaned[["Open", "Close"]].min(axis=1)).all()


def test_feature_order_and_sequence_shape() -> None:
    market = clean_market_data(pd.read_csv(SAMPLE_DATA_PATH))
    features = create_market_features(market)

    assert all(column in features for column in FEATURE_COLUMNS)
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0

    scaled = np.zeros((80, len(FEATURE_COLUMNS)), dtype=np.float32)
    X, y = create_sequences(scaled, look_back=30)

    assert X.shape == (50, 30, 5)
    assert y.shape == (50,)


def test_recursive_forecast_and_replay_are_valid() -> None:
    market = clean_market_data(pd.read_csv(SAMPLE_DATA_PATH))
    model = NumpyBitcoinLSTM(WEIGHTS_PATH)
    scaler = load_scaler(SCALER_PATH)
    look_back = int(load_json(CONFIG_PATH)["look_back"])

    forecast = forecast_future(market, model, scaler, look_back, horizon=7)
    replay = replay_predictions(market, model, scaler, look_back)

    assert len(forecast) == 7
    assert forecast["Date"].is_monotonic_increasing
    assert np.isfinite(forecast["Predicted_Close"]).all()
    assert (forecast["Predicted_Close"] > 0).all()

    assert len(replay) > 100
    assert np.isfinite(replay[["Actual_Close", "Predicted_Close"]]).all().all()
