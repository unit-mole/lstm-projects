from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
from keras import layers
import numpy as np


def set_reproducible_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_lstm_model(
    lookback: int = 12,
    n_features: int = 3,
    lstm_units: int = 16,
    dropout: float = 0.10,
    dense_units: int = 8,
    learning_rate: float = 0.003,
) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(lookback, n_features), name="sequence_input"),
            layers.LSTM(lstm_units, name="seasonal_growth_lstm"),
            layers.Dropout(dropout, name="dropout"),
            layers.Dense(dense_units, activation="relu", name="dense_features"),
            layers.Dense(1, name="growth_forecast"),
        ],
        name="airline_passenger_seasonal_growth_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    epochs: int = 150,
    batch_size: int = 8,
    patience: int = 12,
    verbose: int = 1,
) -> keras.callbacks.History:
    callbacks: list[Any] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=1e-5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(4, patience // 3),
            min_lr=1e-5,
        ),
    ]
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )


def save_model(model: keras.Model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
