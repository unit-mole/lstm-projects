"""Backend-agnostic Keras 3 ConvLSTM model construction and training."""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

from pathlib import Path
from typing import Any

import keras
import numpy as np


def build_convlstm_model(
    input_shape: tuple[int, int, int, int] = (6, 32, 32, 1),
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build the architecture used by the attached trained artifact."""
    inputs = keras.Input(shape=input_shape, name="input_sequence")
    x = keras.layers.ConvLSTM2D(
        32, (3, 3), padding="same", return_sequences=True, activation="tanh"
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ConvLSTM2D(
        32, (3, 3), padding="same", return_sequences=False, activation="tanh"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    outputs = keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    model = keras.Model(inputs, outputs, name="convlstm_next_frame_predictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 15,
    batch_size: int = 32,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> tuple[keras.Model, dict[str, list[float]]]:
    """Train with early stopping and learning-rate reduction."""
    keras.utils.set_random_seed(seed)
    model = build_convlstm_model(tuple(x_train.shape[1:]))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model.save(destination)
    return model, {key: [float(v) for v in values] for key, values in history.history.items()}
