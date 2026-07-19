from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .attention_layer import build_attention_pooling
from .data_generation import generate_ecg_dataset
from .thresholding import optimize_threshold_for_recall


def _require_keras():
    os.environ.setdefault("KERAS_BACKEND", "jax")
    try:
        import keras
    except ImportError as exc:
        raise ImportError(
            "Install requirements.txt before retraining the attention autoencoder."
        ) from exc
    return keras


def build_attention_autoencoder(
    sequence_length: int = 140,
    number_of_features: int = 1,
):
    """Build a true trainable temporal-attention LSTM Autoencoder."""
    keras = _require_keras()
    inputs = keras.Input(
        shape=(sequence_length, number_of_features),
        name="ecg_input",
    )
    encoded_sequence = keras.layers.LSTM(
        64,
        return_sequences=True,
        name="encoder_lstm",
    )(inputs)

    attention_pooling = build_attention_pooling(sequence_length)
    context, _ = attention_pooling(encoded_sequence)
    latent = keras.layers.Dense(
        32,
        activation="tanh",
        name="latent_projection",
    )(context)

    repeated = keras.layers.RepeatVector(
        sequence_length,
        name="repeat_vector",
    )(latent)
    decoded = keras.layers.LSTM(
        32,
        return_sequences=True,
        name="decoder_lstm_1",
    )(repeated)
    decoded = keras.layers.LSTM(
        64,
        return_sequences=True,
        name="decoder_lstm_2",
    )(decoded)
    outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(number_of_features),
        name="reconstruction",
    )(decoded)

    model = keras.Model(
        inputs,
        outputs,
        name="ecg_lstm_autoencoder_attention",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_attention_model(
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    keras = _require_keras()
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    outputs_dir = output_dir / "outputs"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    signals, labels, _ = generate_ecg_dataset(seed=seed)
    train_full, temporary, train_labels, temporary_labels = train_test_split(
        signals,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    validation, test, validation_labels, test_labels = train_test_split(
        temporary,
        temporary_labels,
        test_size=0.50,
        random_state=seed,
        stratify=temporary_labels,
    )
    train_normal = train_full[train_labels == 0]
    validation_normal = validation[validation_labels == 0]

    scaler = StandardScaler()
    scaler.fit(train_normal.reshape(-1, 1))

    def scale(values):
        return scaler.transform(values.reshape(-1, 1)).reshape(values.shape).astype("float32")

    train_scaled = scale(train_normal)
    validation_scaled = scale(validation)
    validation_normal_scaled = validation_scaled[validation_labels == 0]
    test_scaled = scale(test)

    model = build_attention_autoencoder()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]
    history = model.fit(
        train_scaled,
        train_scaled,
        validation_data=(validation_normal_scaled, validation_normal_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    validation_reconstruction = model.predict(validation_scaled, verbose=0)
    validation_errors = np.mean(
        np.abs(validation_scaled - validation_reconstruction),
        axis=(1, 2),
    )
    threshold, validation_f2 = optimize_threshold_for_recall(
        validation_labels,
        validation_errors,
        beta=2.0,
    )

    test_reconstruction = model.predict(test_scaled, verbose=0)
    test_errors = np.mean(
        np.abs(test_scaled - test_reconstruction),
        axis=(1, 2),
    )
    test_predictions = (test_errors >= threshold).astype(int)

    model.save(models_dir / "ecg_lstm_autoencoder_attention_retrained.keras")
    joblib.dump(scaler, models_dir / "ecg_training_scaler.pkl")

    metadata = {
        "sequence_length": 140,
        "number_of_features": 1,
        "threshold": float(threshold),
        "threshold_method": "validation F2 optimization",
        "validation_f2": float(validation_f2),
        "seed": int(seed),
        "test_predictions": int(test_predictions.sum()),
    }
    (models_dir / "attention_retraining_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata
