from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.anomaly_detection import sequence_reconstruction_error
from src.data_preprocessing import (
    DatasetSchema, apply_scaler, clean_sensor_data, fit_training_scaler, split_by_unit,
)
from src.model_evaluation import evaluate_labeled_scores
from src.model_training import TrainingConfig, build_lstm_autoencoder, train_autoencoder
from src.sequence_generation import build_sequences
from src.synthetic_data import generate_predictive_maintenance_data
from src.thresholding import select_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Train the industrial LSTM Autoencoder.")
    parser.add_argument("--data", type=Path, help="Optional CSV with the project schema.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensor_cols = [f"sensor_{i}" for i in range(1, 9)]
    schema = DatasetSchema(sensor_cols=sensor_cols)
    if args.data:
        frame = pd.read_csv(args.data)
    else:
        frame, _ = generate_predictive_maintenance_data()
    frame = clean_sensor_data(frame, schema)
    train_df, validation_df, test_df = split_by_unit(frame, schema)

    scaler = fit_training_scaler(train_df, sensor_cols)
    train_scaled = apply_scaler(train_df, sensor_cols, scaler)
    validation_scaled = apply_scaler(validation_df, sensor_cols, scaler)
    test_scaled = apply_scaler(test_df, sensor_cols, scaler)

    train_batch = build_sequences(train_scaled, schema, window_size=20)
    validation_batch = build_sequences(validation_scaled, schema, window_size=20)
    test_batch = build_sequences(test_scaled, schema, window_size=20)
    healthy_train = train_batch.sequences[train_batch.labels == 0]
    healthy_validation = validation_batch.sequences[validation_batch.labels == 0]

    config = TrainingConfig(epochs=args.epochs)
    model = build_lstm_autoencoder(20, len(sensor_cols), config)
    history = train_autoencoder(model, healthy_train, healthy_validation, config)

    train_reconstruction = model.predict(healthy_train, verbose=0)
    test_reconstruction = model.predict(test_batch.sequences, verbose=0)
    train_errors = sequence_reconstruction_error(healthy_train, train_reconstruction)
    test_errors = sequence_reconstruction_error(test_batch.sequences, test_reconstruction)
    threshold = select_threshold(train_errors)
    evaluation = evaluate_labeled_scores(test_batch.labels, test_errors, threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "lstm_autoencoder_predictive_maintenance.keras")
    joblib.dump(scaler, args.output_dir / "scaler.pkl")
    metadata = {
        "seq_len": 20, "n_features": len(sensor_cols), "sensor_cols": sensor_cols,
        "unit_id_col": schema.unit_id_col, "time_col": schema.time_col,
        "label_col": schema.label_col, "threshold": threshold,
        "threshold_method": "mean + 3 standard deviations of healthy training MAE",
        "seed": 42,
    }
    (args.output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/training_history.json").write_text(json.dumps(history.history, indent=2))
    Path("outputs/model_metrics_retrained.json").write_text(json.dumps(evaluation, indent=2))
    print(f"Saved model artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
