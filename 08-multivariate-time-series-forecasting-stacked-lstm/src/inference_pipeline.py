from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering import add_calendar_features, validate_feature_columns


@dataclass(frozen=True)
class ScalerArtifacts:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    feature_columns: tuple[str, ...]
    sequence_length: int

    @classmethod
    def from_json(cls, path: str | Path) -> "ScalerArtifacts":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float32),
            target_mean=float(payload["target_mean"][0]),
            target_scale=float(payload["target_scale"][0]),
            feature_columns=tuple(payload["feature_cols"]),
            sequence_length=int(payload["seq_len"]),
        )

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - self.feature_mean) / self.feature_scale

    def inverse_target(self, scaled_values: np.ndarray) -> np.ndarray:
        return np.asarray(scaled_values).reshape(-1) * self.target_scale + self.target_mean


def load_keras_model(path: str | Path) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for inference. Install dependencies using "
            "`pip install -r requirements.txt`."
        ) from exc
    return tf.keras.models.load_model(path, compile=False)


def predict_next(model: Any, recent_history: pd.DataFrame, scalers: ScalerArtifacts) -> float:
    frame = add_calendar_features(recent_history)
    validate_feature_columns(frame, scalers.feature_columns)
    if len(frame) < scalers.sequence_length:
        raise ValueError(f"At least {scalers.sequence_length} historical rows are required.")
    window = frame.iloc[-scalers.sequence_length:][list(scalers.feature_columns)].to_numpy()
    x = scalers.transform_features(window)[None, :, :]
    scaled_prediction = np.asarray(model.predict(x, verbose=0)).reshape(-1)
    return float(scalers.inverse_target(scaled_prediction)[0])


def recursive_forecast(
    model: Any,
    history: pd.DataFrame,
    future_exogenous: pd.DataFrame,
    scalers: ScalerArtifacts,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Forecast recursively using user-provided future temperature and humidity.

    The supplied model predicts one hour at a time. Its predicted energy load is
    fed back as the target-history feature for the next step. Future temperature
    and humidity must therefore be supplied or explicitly estimated upstream.
    """
    required_future = {timestamp_column, "temperature", "humidity"}
    missing = required_future.difference(future_exogenous.columns)
    if missing:
        raise ValueError(f"Future exogenous data is missing columns: {sorted(missing)}")

    rolling = add_calendar_features(history).copy().sort_values(timestamp_column)
    if len(rolling) < scalers.sequence_length:
        raise ValueError(f"History needs at least {scalers.sequence_length} rows.")

    future = future_exogenous.copy()
    future[timestamp_column] = pd.to_datetime(future[timestamp_column], errors="raise")
    future = future.sort_values(timestamp_column).reset_index(drop=True)
    forecasts: list[dict[str, float | pd.Timestamp]] = []

    for _, row in future.iterrows():
        prediction = predict_next(model, rolling, scalers)
        timestamp = pd.Timestamp(row[timestamp_column])
        appended = pd.DataFrame([{
            timestamp_column: timestamp,
            "energy_load": prediction,
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
        }])
        appended = add_calendar_features(appended)
        rolling = pd.concat([rolling, appended], ignore_index=True).tail(scalers.sequence_length)
        forecasts.append({
            timestamp_column: timestamp,
            "forecasted_energy_load": prediction,
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
        })

    return pd.DataFrame(forecasts)


def seasonal_future_exogenous(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create a transparent 24-hour seasonal-naive exogenous assumption."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 24:
        raise ValueError("At least 24 historical rows are needed for seasonal assumptions.")
    inferred_step = frame["timestamp"].diff().dropna().median()
    if pd.isna(inferred_step) or inferred_step <= pd.Timedelta(0):
        inferred_step = pd.Timedelta(hours=1)

    rows = []
    for step in range(horizon):
        source = frame.iloc[-24 + (step % 24)]
        rows.append({
            "timestamp": frame["timestamp"].iloc[-1] + inferred_step * (step + 1),
            "temperature": float(source["temperature"]),
            "humidity": float(source["humidity"]),
        })
    return pd.DataFrame(rows)
