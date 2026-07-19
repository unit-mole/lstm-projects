from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CONFIG, MODEL_DIR, OUTPUT_DIR, ForecastConfig
from src.data_preprocessing import chronological_split, load_time_series_csv, prepare_time_series
from src.feature_engineering import add_calendar_features
from src.model_evaluation import naive_previous_value, regression_metrics
from src.model_training import build_stacked_lstm, train_model
from src.sequence_generation import build_supervised_sequences


def run_training_pipeline(
    data_path: str | Path,
    config: ForecastConfig = DEFAULT_CONFIG,
    epochs: int = 20,
    batch_size: int = 64,
) -> dict:
    """Train, evaluate and persist a leakage-controlled Stacked LSTM pipeline."""
    random.seed(config.seed)
    np.random.seed(config.seed)

    raw = load_time_series_csv(data_path)
    clean, quality_report = prepare_time_series(
        raw,
        timestamp_column=config.timestamp_column,
        required_numeric_columns=(config.target_column, "temperature", "humidity"),
    )
    featured = add_calendar_features(clean, config.timestamp_column)
    train_frame, validation_frame, test_frame = chronological_split(
        featured, config.train_ratio, config.validation_ratio
    )

    # Scalers are fitted on training data only.
    feature_scaler = StandardScaler().fit(train_frame[list(config.feature_columns)])
    target_scaler = StandardScaler().fit(train_frame[[config.target_column]])

    def sequence_partition(frame):
        scaled_features = feature_scaler.transform(frame[list(config.feature_columns)])
        scaled_target = target_scaler.transform(frame[[config.target_column]]).reshape(-1)
        return build_supervised_sequences(
            scaled_features,
            scaled_target,
            sequence_length=config.sequence_length,
            forecast_horizon=config.forecast_horizon,
        )

    x_train, y_train = sequence_partition(train_frame)
    x_validation, y_validation = sequence_partition(validation_frame)
    x_test, y_test = sequence_partition(test_frame)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "stacked_lstm_energy.keras"
    model = build_stacked_lstm((x_train.shape[1], x_train.shape[2]), config.forecast_horizon)
    history = train_model(
        model, x_train, y_train, x_validation, y_validation, model_path, epochs, batch_size
    )

    validation_scaled = model.predict(x_validation, verbose=0).reshape(-1)
    test_scaled = model.predict(x_test, verbose=0).reshape(-1)
    validation_actual = target_scaler.inverse_transform(y_validation.reshape(-1, 1)).reshape(-1)
    test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    validation_prediction = target_scaler.inverse_transform(validation_scaled.reshape(-1, 1)).reshape(-1)
    test_prediction = target_scaler.inverse_transform(test_scaled.reshape(-1, 1)).reshape(-1)

    metrics = {
        "stacked_lstm_validation": regression_metrics(validation_actual, validation_prediction),
        "stacked_lstm_test": regression_metrics(test_actual, test_prediction),
        "naive_test": regression_metrics(
            test_actual,
            naive_previous_value(test_frame[config.target_column].to_numpy(), config.sequence_length),
        ),
    }
    (OUTPUT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    scaler_payload = {
        "feature_mean": feature_scaler.mean_.tolist(),
        "feature_scale": feature_scaler.scale_.tolist(),
        "target_mean": target_scaler.mean_.tolist(),
        "target_scale": target_scaler.scale_.tolist(),
        "feature_cols": list(config.feature_columns),
        "seq_len": config.sequence_length,
    }
    (MODEL_DIR / "scalers.json").write_text(json.dumps(scaler_payload, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "training_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "quality_report": quality_report.__dict__,
        "model_path": str(model_path),
        "input_shape": list(x_train.shape[1:]),
    }
