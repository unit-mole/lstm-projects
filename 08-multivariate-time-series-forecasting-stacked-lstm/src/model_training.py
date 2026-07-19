from __future__ import annotations

from pathlib import Path
from typing import Any


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for model training. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return tf


def build_stacked_lstm(input_shape: tuple[int, int], forecast_horizon: int = 1):
    """Build the same 34,529-parameter architecture as the supplied model artifact."""
    tf = _tensorflow()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(forecast_horizon),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_callbacks(model_path: str | Path):
    tf = _tensorflow()
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss", save_best_only=True
        ),
    ]


def train_model(
    model,
    x_train,
    y_train,
    x_validation,
    y_validation,
    model_path: str | Path,
    epochs: int = 20,
    batch_size: int = 64,
):
    """Train without shuffling so temporal sample order remains explicit."""
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=build_callbacks(model_path),
        shuffle=False,
        verbose=1,
    )
    model.save(model_path)
    return history
