from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import FEATURE_COLUMNS, MODELS_DIR, OUTPUTS_DIR
from .data_preprocessing import clean_market_data, fetch_optional_yfinance, load_csv, validate_history_length
from .feature_engineering import create_market_features
from .model_evaluation import regression_metrics
from .sequence_generation import create_sequences


def build_lstm_model(
    input_shape: tuple[int, int],
    units_1: int = 64,
    units_2: int = 32,
    dropout: float = 0.20,
    learning_rate: float = 0.001,
):
    """Build the training model. Keras is imported lazily to keep inference lightweight."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import keras

    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.LSTM(units_1, return_sequences=True, name="lstm_1")(inputs)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    x = keras.layers.LSTM(units_2, name="lstm_2")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense_hidden")(x)
    outputs = keras.layers.Dense(1, name="close_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="bitcoin_price_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def _inverse_close(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    placeholder = np.zeros((len(values), len(FEATURE_COLUMNS)))
    placeholder[:, 0] = np.asarray(values).reshape(-1)
    return scaler.inverse_transform(placeholder)[:, 0]


def export_numpy_weights(model, output_path: str | Path) -> None:
    """Export a fixed two-LSTM/two-Dense model for backend-free cloud inference."""
    lstm_1 = model.get_layer("lstm_1").get_weights()
    lstm_2 = model.get_layer("lstm_2").get_weights()
    dense_1 = model.get_layer("dense_hidden").get_weights()
    dense_2 = model.get_layer("close_output").get_weights()
    np.savez_compressed(
        output_path,
        lstm1_kernel=lstm_1[0],
        lstm1_recurrent_kernel=lstm_1[1],
        lstm1_bias=lstm_1[2],
        lstm2_kernel=lstm_2[0],
        lstm2_recurrent_kernel=lstm_2[1],
        lstm2_bias=lstm_2[2],
        dense1_kernel=dense_1[0],
        dense1_bias=dense_1[1],
        dense2_kernel=dense_2[0],
        dense2_bias=dense_2[1],
    )


def train_project(
    csv_path: str | Path | None = None,
    ticker: str = "BTC-USD",
    period: str = "8y",
    look_back: int = 30,
    epochs: int = 80,
    batch_size: int = 32,
) -> dict[str, float]:
    """
    Retrain with strict chronological train/validation/test periods.

    The scaler is fitted only on the training feature rows. Validation controls
    early stopping; the final test period is untouched until evaluation.
    """
    if csv_path:
        market = clean_market_data(load_csv(csv_path))
        data_source = str(csv_path)
    else:
        market = fetch_optional_yfinance(ticker=ticker, period=period)
        data_source = f"yfinance:{ticker}:{period}"

    validate_history_length(market, minimum_rows=max(240, look_back + 90))
    features = create_market_features(market)

    n_rows = len(features)
    train_end = int(n_rows * 0.70)
    validation_end = int(n_rows * 0.85)

    scaler = MinMaxScaler()
    scaler.fit(features.loc[: train_end - 1, FEATURE_COLUMNS])
    scaled = scaler.transform(features[FEATURE_COLUMNS])

    X, y = create_sequences(scaled, look_back=look_back, target_col_index=0)
    target_indices = np.arange(look_back, len(features))

    train_mask = target_indices < train_end
    validation_mask = (target_indices >= train_end) & (target_indices < validation_end)
    test_mask = target_indices >= validation_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[validation_mask], y[validation_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = build_lstm_model((look_back, len(FEATURE_COLUMNS)))
    import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    pred_scaled = np.asarray(model.predict(X_test, verbose=0)).reshape(-1)
    actual = _inverse_close(y_test, scaler)
    predicted = _inverse_close(pred_scaled, scaler)
    metrics = regression_metrics(actual, predicted)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "bitcoin_lstm_model.keras")
    export_numpy_weights(model, MODELS_DIR / "bitcoin_lstm_weights.npz")
    with open(MODELS_DIR / "bitcoin_scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    metadata = {
        "project": "Bitcoin Price Prediction using LSTM",
        "data_source": data_source,
        "target_column": "Close",
        "feature_columns": FEATURE_COLUMNS,
        "look_back": look_back,
        "forecast_horizons": [1, 7, 14, 30],
        "split_strategy": "chronological_70_15_15",
        "scaler_fit_scope": "training_rows_only",
        "train_period": [
            str(features["Date"].iloc[0].date()),
            str(features["Date"].iloc[train_end - 1].date()),
        ],
        "validation_period": [
            str(features["Date"].iloc[train_end].date()),
            str(features["Date"].iloc[validation_end - 1].date()),
        ],
        "test_period": [
            str(features["Date"].iloc[validation_end].date()),
            str(features["Date"].iloc[-1].date()),
        ],
        "test_metrics": metrics,
    }
    (MODELS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    prediction_dates = features["Date"].iloc[target_indices[test_mask]].reset_index(drop=True)
    pd.DataFrame(
        {
            "Date": prediction_dates,
            "Actual_Close": actual,
            "Predicted_Close": predicted,
            "Residual": actual - predicted,
        }
    ).to_csv(OUTPUTS_DIR / "test_predictions.csv", index=False)
    pd.DataFrame(history.history).to_csv(OUTPUTS_DIR / "training_history.csv", index=False)
    (OUTPUTS_DIR / "model_metrics.json").write_text(
        json.dumps({"strict_retraining_test_metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    return metrics
