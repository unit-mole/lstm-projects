from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingConfig:
    window_size: int = 20
    encoder_units: tuple[int, int] = (64, 32)
    decoder_units: tuple[int, int] = (32, 64)
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 20
    patience: int = 5


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for training. Install requirements-dev.txt. "
            "The hosted app does not need TensorFlow because it has a portable NumPy backend."
        ) from exc
    return tf


def build_lstm_autoencoder(
    sequence_length: int,
    n_features: int,
    config: TrainingConfig | None = None,
):
    """Build the encoder–bottleneck–decoder architecture used by the artifact."""
    tf = _tensorflow()
    cfg = config or TrainingConfig(window_size=sequence_length)
    inputs = tf.keras.Input(shape=(sequence_length, n_features), name="sensor_sequence")
    x = tf.keras.layers.LSTM(cfg.encoder_units[0], return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(cfg.encoder_units[1], return_sequences=False, name="latent_vector")(x)
    x = tf.keras.layers.RepeatVector(sequence_length)(x)
    x = tf.keras.layers.LSTM(cfg.decoder_units[0], return_sequences=True)(x)
    x = tf.keras.layers.LSTM(cfg.decoder_units[1], return_sequences=True)(x)
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_features), name="reconstructed_sequence"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="industrial_lstm_autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_autoencoder(
    model,
    healthy_train_sequences: np.ndarray,
    healthy_validation_sequences: np.ndarray,
    config: TrainingConfig | None = None,
):
    """Train on normal sequences only and restore the best validation weights."""
    tf = _tensorflow()
    cfg = config or TrainingConfig(window_size=healthy_train_sequences.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    return model.fit(
        healthy_train_sequences,
        healthy_train_sequences,
        validation_data=(healthy_validation_sequences, healthy_validation_sequences),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
