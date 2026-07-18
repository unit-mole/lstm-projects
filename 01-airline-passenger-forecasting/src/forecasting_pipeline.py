from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import joblib
import keras
import numpy as np
import pandas as pd

from .data_preprocessing import prepare_monthly_series
from .feature_engineering import cyclical_month_features, log_passenger_levels, seasonal_log_difference
from .model_evaluation import calculate_metrics
from .sequence_generation import build_sequence_dataset


def load_artifacts(
    model_path: str | Path,
    scaler_path: str | Path,
    metadata_path: str | Path,
):
    model = keras.saving.load_model(model_path, compile=False)
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
