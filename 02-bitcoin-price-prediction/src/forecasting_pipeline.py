from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .cloud_inference import NumpyBitcoinLSTM
from .feature_engineering import FEATURE_COLUMNS, create_market_features
from .sequence_generation import create_sequences


def load_scaler(path: str | Path):
    with open(path, "rb") as file:
        return pickle.load(file)


def inverse_close(scaled_close: np.ndarray, scaler) -> np.ndarray:
    values = np.asarray(scaled_close, dtype=float).reshape(-1)
    placeholder = np.zeros((len(values), len(FEATURE_COLUMNS)), dtype=float)
    placeholder[:, 0] = values
    return scaler.inverse_transform(placeholder)[:, 0]


def replay_predictions(
    market_df: pd.DataFrame,
    model: NumpyBitcoinLSTM,
    scaler,
    look_back: int,
) -> pd.DataFrame:
    feature_df = create_market_features(market_df)
    scaled = scaler.transform(feature_df[FEATURE_COLUMNS].to_numpy(dtype=float))
    X, y = create_sequences(scaled, look_back=look_back, target_col_index=0)
    pred_scaled = model.predict(X).reshape(-1)

    return pd.DataFrame(
        {
            "Date": feature_df["Date"].iloc[look_back:].reset_index(drop=True),
            "Actual_Close": inverse_close(y, scaler),
            "Predicted_Close": inverse_close(pred_scaled, scaler),
        }
    )


def forecast_future(
    market_df: pd.DataFrame,
    model: NumpyBitcoinLSTM,
    scaler,
    look_back: int,
    horizon: int,
) -> pd.DataFrame:
    """
    Recursively forecast Close while recalculating SMA and return features.
    Volume is carried forward as the recent 30-day median.
    """
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one day.")

    history = market_df.copy().sort_values("Date").reset_index(drop=True)
    feature_df = create_market_features(history)
    if len(feature_df) < look_back:
        raise ValueError(f"At least {look_back + 30} prepared rows are required.")

    recent_volume = float(history["Volume"].tail(30).median())

    forecasts: list[dict[str, object]] = []
    for _ in range(horizon):
        feature_df = create_market_features(history)
        matrix = feature_df[FEATURE_COLUMNS].tail(look_back).to_numpy(dtype=float)
        scaled_window = scaler.transform(matrix).astype(np.float32)
        pred_scaled = model.predict(scaled_window).reshape(-1)[0]
        predicted_close = float(inverse_close(np.array([pred_scaled]), scaler)[0])

        previous_close = float(history["Close"].iloc[-1])
        next_date = pd.Timestamp(history["Date"].iloc[-1]) + pd.Timedelta(days=1)
        open_price = previous_close
        high_price = max(open_price, predicted_close)
        low_price = min(open_price, predicted_close)

        next_row = {
            "Date": next_date,
            "Open": open_price,
            "High": high_price,
            "Low": low_price,
            "Close": predicted_close,
            "Volume": recent_volume,
        }
        history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
        forecasts.append(
            {
                "Date": next_date,
                "Predicted_Close": predicted_close,
                "Change_From_Previous_Day_Percent": (
                    (predicted_close / previous_close - 1.0) * 100.0
                ),
            }
        )

    return pd.DataFrame(forecasts)


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
