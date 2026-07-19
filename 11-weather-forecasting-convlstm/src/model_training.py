from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import MODEL_DIR, OUTPUT_DIR
from src.data_preprocessing import split_independent_sequences
from src.model_evaluation import map_mae, map_rmse
from src.synthetic_weather import generate_weather_sequences


def build_convlstm(input_shape: tuple[int, int, int, int]):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True, activation="tanh")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(32, (3, 3), padding="same", return_sequences=False, activation="tanh")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model


def train(n_samples: int = 2200, epochs: int = 15, batch_size: int = 32, seed: int = 42) -> None:
    import tensorflow as tf
    X, future = generate_weather_sequences(n_samples=n_samples, future_frames=1, seed=seed)
    y = future[:, 0]
    X_train, y_train, X_val, y_val, X_test, y_test = split_independent_sequences(X, y)
    model = build_convlstm(X_train.shape[1:])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "convlstm_weather_forecast_retrained.keras")
    predictions = model.predict(X_test, verbose=0)
    metrics = {"test_mae": map_mae(y_test, predictions), "test_rmse": map_rmse(y_test, predictions)}
    (OUTPUT_DIR / "retrained_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "retrained_history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ConvLSTM weather forecasting model")
    parser.add_argument("--samples", type=int, default=2200)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.samples, args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
