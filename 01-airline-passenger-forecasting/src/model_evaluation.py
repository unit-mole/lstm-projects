from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .feature_engineering import log_passenger_levels
from .sequence_generation import SequenceDataset


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    mape: float
    r2: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> ForecastMetrics:
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
    return ForecastMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
        mape=mape,
        r2=float(r2_score(actual, predicted)),
    )


def inverse_growth_predictions(
    standardized_growth: np.ndarray,
    target_indices: np.ndarray,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    seasonal_period: int = 12,
) -> np.ndarray:
    growth = scaler.inverse_transform(np.asarray(standardized_growth).reshape(-1, 1)).ravel()
    log_levels = log_passenger_levels(frame["Passengers"])
    predicted_log = log_levels[target_indices - seasonal_period] + growth
    return np.expm1(predicted_log)


def predict_sequence_subset(
    model,
    dataset: SequenceDataset,
    mask: np.ndarray,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    seasonal_period: int = 12,
) -> pd.DataFrame:
    indices = dataset.target_indices[mask]
    standardized_predictions = model.predict(dataset.X[mask], verbose=0).ravel()
    predicted = inverse_growth_predictions(
        standardized_predictions, indices, frame, scaler, seasonal_period
    )
    result = frame.iloc[indices][["Month", "Passengers"]].copy()
    result = result.rename(columns={"Passengers": "Actual"})
    result["Predicted"] = predicted
    result["Residual"] = result["Actual"] - result["Predicted"]
    return result.reset_index(drop=True)


def baseline_predictions(
    frame: pd.DataFrame,
    target_indices: np.ndarray,
    fit_end: int,
    seasonal_period: int = 12,
) -> dict[str, np.ndarray]:
    values = frame["Passengers"].to_numpy(dtype=float)
    naive = values[target_indices - 1]
    seasonal_naive = values[target_indices - seasonal_period]
    moving_average = np.asarray(
        [values[index - seasonal_period:index].mean() for index in target_indices],
        dtype=float,
    )

    trend_model = LinearRegression()
    trend_model.fit(np.arange(fit_end).reshape(-1, 1), values[:fit_end])
    linear_trend = trend_model.predict(target_indices.reshape(-1, 1))

    return {
        "Naive (previous month)": naive,
        "Seasonal naive (previous year)": seasonal_naive,
        "12-month moving average": moving_average,
        "Linear trend": linear_trend,
    }


def comparison_table(
    actual: np.ndarray,
    lstm_predictions: np.ndarray,
    baselines: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for name, predictions in baselines.items():
        rows.append({"Model": name, **calculate_metrics(actual, predictions).to_dict()})
    rows.append({"Model": "Seasonally adjusted LSTM", **calculate_metrics(actual, lstm_predictions).to_dict()})
    result = pd.DataFrame(rows)
    return result.sort_values("rmse").reset_index(drop=True)
