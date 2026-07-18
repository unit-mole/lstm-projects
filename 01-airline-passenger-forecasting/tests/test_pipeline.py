from pathlib import Path

import pandas as pd

from src.data_preprocessing import load_and_prepare
from src.feature_engineering import seasonal_log_difference
from src.sequence_generation import build_sequence_dataset, fit_growth_scaler


def test_sample_data_is_monthly_and_complete():
    project_root = Path(__file__).resolve().parents[1]
    frame, notes = load_and_prepare(project_root / "data" / "airline_passengers_sample.csv")
    assert len(frame) == 144
    assert frame["Month"].is_monotonic_increasing
    assert frame["Month"].nunique() == len(frame)
    assert frame["Passengers"].notna().all()


def test_training_scaler_and_sequences_do_not_require_future_fit_data():
    project_root = Path(__file__).resolve().parents[1]
    frame, _ = load_and_prepare(project_root / "data" / "airline_passengers_sample.csv")
    scaler = fit_growth_scaler(frame, train_end=96, seasonal_period=12)
    dataset = build_sequence_dataset(frame, scaler, lookback=12, seasonal_period=12)
    assert dataset.X.shape[1:] == (12, 3)
    assert dataset.target_indices.min() == 24
    assert dataset.target_indices.max() == 143
