from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data_preprocessing import prepare_monthly_series
from .feature_engineering import cyclical_month_features, log_passenger_levels
from .model_evaluation import calculate_metrics
from .sequence_generation import build_sequence_dataset


class NumpyLSTMInferenceModel:
    """Backend-free inference implementation for the packaged Keras LSTM.

    The saved NPZ file contains the trained weights from the original Keras model.
    This class reproduces the Keras LSTM forward pass with NumPy, so the deployed
    Streamlit app does not depend on a JAX or TensorFlow runtime.
    """

    def __init__(
        self,
        lstm_kernel: np.ndarray,
        lstm_recurrent_kernel: np.ndarray,
        lstm_bias: np.ndarray,
        dense_kernel: np.ndarray,
        dense_bias: np.ndarray,
        output_kernel: np.ndarray,
        output_bias: np.ndarray,
    ) -> None:
        self.lstm_kernel = np.asarray(lstm_kernel, dtype=np.float32)
        self.lstm_recurrent_kernel = np.asarray(lstm_recurrent_kernel, dtype=np.float32)
        self.lstm_bias = np.asarray(lstm_bias, dtype=np.float32)
        self.dense_kernel = np.asarray(dense_kernel, dtype=np.float32)
        self.dense_bias = np.asarray(dense_bias, dtype=np.float32)
        self.output_kernel = np.asarray(output_kernel, dtype=np.float32)
        self.output_bias = np.asarray(output_bias, dtype=np.float32)

        if self.lstm_kernel.ndim != 2 or self.lstm_kernel.shape[1] % 4 != 0:
            raise ValueError("Invalid LSTM kernel stored in the NumPy model artifact.")
        self.units = self.lstm_kernel.shape[1] // 4

    @classmethod
    def load(cls, path: str | Path) -> "NumpyLSTMInferenceModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NumPy LSTM artifact was not found: {path}")

        with np.load(path, allow_pickle=False) as weights:
            required = {
                "lstm_kernel",
                "lstm_recurrent_kernel",
                "lstm_bias",
                "dense_kernel",
                "dense_bias",
                "output_kernel",
                "output_bias",
            }
            missing = sorted(required.difference(weights.files))
            if missing:
                raise ValueError(
                    "The NumPy LSTM artifact is incomplete. Missing arrays: "
                    + ", ".join(missing)
                )
            return cls(**{name: weights[name] for name in required})

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-values))

    def predict(self, inputs: np.ndarray, verbose: int = 0, **_: Any) -> np.ndarray:
        del verbose
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(
                "LSTM inference expects a 3D array shaped "
                "(batch, timesteps, features)."
            )
        if x.shape[-1] != self.lstm_kernel.shape[0]:
            raise ValueError(
                f"Expected {self.lstm_kernel.shape[0]} input features, "
                f"but received {x.shape[-1]}."
            )

        batch_size = x.shape[0]
        hidden = np.zeros((batch_size, self.units), dtype=np.float32)
        cell = np.zeros_like(hidden)

        # Keras LSTM gate order is input, forget, cell candidate, output.
        for timestep in range(x.shape[1]):
            gates = (
                x[:, timestep, :] @ self.lstm_kernel
                + hidden @ self.lstm_recurrent_kernel
                + self.lstm_bias
            )
            input_gate, forget_gate, candidate, output_gate = np.split(
                gates, 4, axis=-1
            )
            input_gate = self._sigmoid(input_gate)
            forget_gate = self._sigmoid(forget_gate)
            candidate = np.tanh(candidate)
            output_gate = self._sigmoid(output_gate)

            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * np.tanh(cell)

        dense = np.maximum(hidden @ self.dense_kernel + self.dense_bias, 0.0)
        output = dense @ self.output_kernel + self.output_bias
        return np.asarray(output, dtype=np.float32)


def _load_keras_model(model_path: Path):
    """Fallback loader for local retraining environments with Keras installed."""
    try:
        import os

        os.environ.setdefault("KERAS_BACKEND", "jax")
        import keras
    except Exception as exc:  # pragma: no cover - deployment uses the NPZ path.
        raise RuntimeError(
            "The NumPy inference artifact is missing and Keras could not be loaded. "
            "Restore models/airline_passenger_lstm_weights.npz or install the "
            "project training dependencies."
        ) from exc
    return keras.saving.load_model(model_path, compile=False)


def load_artifacts(
    model_path: str | Path,
    scaler_path: str | Path,
    metadata_path: str | Path,
):
    model_path = Path(model_path)
    numpy_model_path = model_path.with_name("airline_passenger_lstm_weights.npz")

    if numpy_model_path.exists():
        model = NumpyLSTMInferenceModel.load(numpy_model_path)
    else:
        model = _load_keras_model(model_path)

    scaler = joblib.load(scaler_path)
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return model, scaler, metadata


def historical_one_step_predictions(
    frame: pd.DataFrame,
    model,
    scaler,
    lookback: int = 12,
    seasonal_period: int = 12,
) -> pd.DataFrame:
    dataset = build_sequence_dataset(frame, scaler, lookback, seasonal_period)
    standardized = model.predict(dataset.X, verbose=0).ravel()
    growth = scaler.inverse_transform(standardized.reshape(-1, 1)).ravel()
    log_levels = log_passenger_levels(frame["Passengers"])
    predictions = np.expm1(log_levels[dataset.target_indices - seasonal_period] + growth)
    result = frame.iloc[dataset.target_indices][["Month", "Passengers"]].copy()
    result = result.rename(columns={"Passengers": "Actual"})
    result["Predicted"] = predictions
    result["Residual"] = result["Actual"] - result["Predicted"]
    return result.reset_index(drop=True)


def recursive_forecast(
    frame: pd.DataFrame,
    model,
    scaler,
    horizon: int,
    lookback: int = 12,
    seasonal_period: int = 12,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one month.")
    required = seasonal_period + lookback
    if len(frame) < required:
        raise ValueError(f"At least {required} monthly observations are required.")

    log_levels = list(log_passenger_levels(frame["Passengers"]))
    dates = list(pd.to_datetime(frame["Month"]))

    for _ in range(horizon):
        differences = np.full(len(log_levels), np.nan, dtype=np.float32)
        levels = np.asarray(log_levels, dtype=np.float32)
        differences[seasonal_period:] = levels[seasonal_period:] - levels[:-seasonal_period]
        latest_differences = differences[-lookback:]
        if np.isnan(latest_differences).any():
            raise ValueError("Insufficient seasonal history to create the forecast sequence.")

        standardized = scaler.transform(latest_differences.reshape(-1, 1)).ravel()
        input_dates = pd.DatetimeIndex(dates[-lookback:])
        month_sin, month_cos = cyclical_month_features(input_dates.month)
        features = np.column_stack([standardized, month_sin, month_cos]).astype(np.float32)
        predicted_standardized = float(model.predict(features[None, ...], verbose=0)[0][0])
        predicted_growth = float(
            scaler.inverse_transform([[predicted_standardized]])[0][0]
        )
        next_log_level = log_levels[-seasonal_period] + predicted_growth
        next_date = dates[-1] + pd.offsets.MonthBegin(1)
        log_levels.append(next_log_level)
        dates.append(next_date)

    forecast_dates = dates[-horizon:]
    forecast_values = np.expm1(np.asarray(log_levels[-horizon:], dtype=float))
    return pd.DataFrame(
        {
            "Month": pd.to_datetime(forecast_dates),
            "Forecasted_Passengers": np.maximum(forecast_values, 0.0),
        }
    )


def summarize_forecast(forecast: pd.DataFrame) -> dict[str, str | float]:
    first_value = float(forecast["Forecasted_Passengers"].iloc[0])
    last_value = float(forecast["Forecasted_Passengers"].iloc[-1])
    average_value = float(forecast["Forecasted_Passengers"].mean())
    change_pct = ((last_value - first_value) / first_value * 100.0) if first_value else 0.0
    if change_pct > 2:
        direction = "increasing"
    elif change_pct < -2:
        direction = "decreasing"
    else:
        direction = "broadly stable"
    return {
        "forecast_horizon_months": int(len(forecast)),
        "average_forecast": average_value,
        "ending_forecast": last_value,
        "change_percent": change_pct,
        "trend_direction": direction,
        "business_interpretation": (
            f"Passenger demand is forecast to be {direction} across the selected horizon. "
            "The seasonal monthly pattern should be considered when planning capacity, "
            "schedules, staffing, and revenue targets."
        ),
    }


def evaluate_history(frame: pd.DataFrame, model, scaler, lookback: int = 12, seasonal_period: int = 12):
    predictions = historical_one_step_predictions(frame, model, scaler, lookback, seasonal_period)
    metrics = calculate_metrics(predictions["Actual"], predictions["Predicted"])
    return predictions, metrics
