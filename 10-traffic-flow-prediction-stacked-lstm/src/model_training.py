"""Optional reproducible TensorFlow training pipeline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from .data_preprocessing import load_traffic_csv
from .data_preprocessing import prepare_traffic_data
from .sequence_generation import build_sequences
from .synthetic_data import generate_traffic_data


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for retraining. "
            "Install requirements-dev.txt first."
        ) from exc
    return tf


def build_stacked_lstm(input_shape: tuple[int, int]):
    """Build the architecture used by the supplied notebook."""
    tf = _tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train(
    data_path: str | None = None,
    output_directory: str | Path = "models",
    sequence_length: int = 24,
    epochs: int = 20,
    batch_size: int = 64,
    seed: int = 42,
) -> dict:
    """Train using a chronological 70/15/15 split and training-only scalers."""
    tf = _tensorflow()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    raw = (
        load_traffic_csv(data_path)
        if data_path
        else generate_traffic_data(seed=seed)
    )
    prepared = prepare_traffic_data(raw)

    feature_columns = [
        "congestion_index",
        "vehicle_count",
        "avg_speed",
        "occupancy",
        "weather_severity",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "weekend",
    ]
    target_column = "congestion_index"

    row_count = len(prepared)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    train_frame = prepared.iloc[:train_end]
    validation_frame = prepared.iloc[train_end:validation_end]
    test_frame = prepared.iloc[validation_end:]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(
        train_frame[feature_columns]
    )
    validation_features = feature_scaler.transform(
        validation_frame[feature_columns]
    )
    test_features = feature_scaler.transform(test_frame[feature_columns])

    train_target = target_scaler.fit_transform(
        train_frame[[target_column]]
    ).reshape(-1)
    validation_target = target_scaler.transform(
        validation_frame[[target_column]]
    ).reshape(-1)
    test_target = target_scaler.transform(
        test_frame[[target_column]]
    ).reshape(-1)

    X_train, y_train = build_sequences(
        train_features,
        train_target,
        sequence_length,
    )
    X_validation, y_validation = build_sequences(
        validation_features,
        validation_target,
        sequence_length,
    )
    X_test, y_test = build_sequences(
        test_features,
        test_target,
        sequence_length,
    )

    model = build_stacked_lstm(
        (X_train.shape[1], X_train.shape[2])
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    predictions_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    actual = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).reshape(-1)
    predicted = target_scaler.inverse_transform(
        predictions_scaled
    ).reshape(-1)

    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save(output_path / "stacked_lstm_traffic.keras")
    scaler_payload = {
        "feature_mean": feature_scaler.mean_.tolist(),
        "feature_scale": feature_scaler.scale_.tolist(),
        "target_mean": target_scaler.mean_.tolist(),
        "target_scale": target_scaler.scale_.tolist(),
        "feature_cols": feature_columns,
        "seq_len": sequence_length,
    }
    (output_path / "scalers.json").write_text(
        json.dumps(scaler_payload, indent=2),
        encoding="utf-8",
    )

    history_path = output_path.parent / "outputs"
    history_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(
        history_path / "training_history.csv",
        index=False,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--output-directory", default="models")
    arguments = parser.parse_args()
    metrics = train(
        data_path=arguments.data,
        output_directory=arguments.output_directory,
        sequence_length=arguments.sequence_length,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
