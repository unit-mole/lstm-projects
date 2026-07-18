from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .feature_engineering import cyclical_month_features, seasonal_log_difference


@dataclass(frozen=True)
class SequenceDataset:
    X: np.ndarray
    y: np.ndarray
    target_indices: np.ndarray
    feature_names: tuple[str, ...]


def fit_growth_scaler(
    frame: pd.DataFrame,
    train_end: int,
    seasonal_period: int = 12,
) -> StandardScaler:
    """Fit only on seasonal-growth observations whose targets are in training."""
    differences = seasonal_log_difference(frame["Passengers"], seasonal_period)
    training_values = differences[seasonal_period:train_end]
    training_values = training_values[~np.isnan(training_values)]
    if training_values.size < 12:
        raise ValueError("Not enough training data to fit the seasonal-growth scaler.")
    return StandardScaler().fit(training_values.reshape(-1, 1))


def build_sequence_dataset(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    lookback: int = 12,
    seasonal_period: int = 12,
) -> SequenceDataset:
    """
    Build LSTM sequences from standardized year-over-year log growth plus
    cyclical month features.
    """
    differences = seasonal_log_difference(frame["Passengers"], seasonal_period)
    standardized = np.full_like(differences, np.nan, dtype=np.float32)
    valid = ~np.isnan(differences)
    standardized[valid] = scaler.transform(differences[valid].reshape(-1, 1)).ravel()

    X, y, target_indices = [], [], []
    first_target = seasonal_period + lookback
    for target_index in range(first_target, len(frame)):
        input_indices = np.arange(target_index - lookback, target_index)
        month_sin, month_cos = cyclical_month_features(
            frame.iloc[input_indices]["MonthNumber"].to_numpy()
        )
        features = np.column_stack(
            [standardized[input_indices], month_sin, month_cos]
        ).astype(np.float32)
        if np.isnan(features).any() or np.isnan(standardized[target_index]):
            continue
        X.append(features)
        y.append(standardized[target_index])
        target_indices.append(target_index)

    if not X:
        raise ValueError("No valid sequences could be created from the supplied data.")

    return SequenceDataset(
        X=np.asarray(X, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        target_indices=np.asarray(target_indices, dtype=int),
        feature_names=("standardized_seasonal_log_growth", "month_sin", "month_cos"),
    )


def chronological_masks(
    target_indices: np.ndarray,
    train_end: int,
    validation_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = target_indices < train_end
    validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
    test_mask = target_indices >= validation_end
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("Each chronological split must contain at least one sequence.")
    return train_mask, validation_mask, test_mask
