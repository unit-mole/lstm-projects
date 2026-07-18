from __future__ import annotations

import io
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import joblib
import pandas as pd

from src.config import (
    BEST_CONFIG_PATH,
    LOOKBACK,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    SAMPLE_DATA_PATH,
    SCALER_PATH,
    SEASONAL_PERIOD,
)
from src.data_preprocessing import load_and_prepare
from src.forecasting_pipeline import recursive_forecast, summarize_forecast
from src.model_evaluation import (
    baseline_predictions,
    calculate_metrics,
    comparison_table,
    predict_sequence_subset,
)
from src.model_training import build_lstm_model, save_model, set_reproducible_seed, train_model
from src.sequence_generation import (
    build_sequence_dataset,
    chronological_masks,
    fit_growth_scaler,
)
from src.visualization import (
    plot_actual_vs_predicted,
    plot_baseline_comparison,
    plot_forecast,
    plot_passenger_trend,
    plot_residuals,
    plot_seasonal_pattern,
    plot_training_curve,
)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(RANDOM_SEED)

    frame, preprocessing_notes = load_and_prepare(SAMPLE_DATA_PATH)

    # For the 144-month AirPassengers series: 96 train, 24 validation, 24 test.
    train_end = 96
    validation_end = 120

    scaler = fit_growth_scaler(frame, train_end, SEASONAL_PERIOD)
    dataset = build_sequence_dataset(frame, scaler, LOOKBACK, SEASONAL_PERIOD)
    train_mask, validation_mask, test_mask = chronological_masks(
        dataset.target_indices, train_end, validation_end
    )

    config = {
        "lookback": LOOKBACK,
        "seasonal_period": SEASONAL_PERIOD,
        "effective_raw_history_months": LOOKBACK + SEASONAL_PERIOD,
        "n_features": dataset.X.shape[-1],
        "feature_names": list(dataset.feature_names),
        "lstm_units": 16,
        "dropout": 0.10,
        "dense_units": 8,
        "learning_rate": 0.003,
        "batch_size": 8,
        "maximum_epochs": 150,
        "early_stopping_patience": 12,
        "random_seed": RANDOM_SEED,
    }

    model = build_lstm_model(
        lookback=config["lookback"],
        n_features=config["n_features"],
        lstm_units=config["lstm_units"],
        dropout=config["dropout"],
        dense_units=config["dense_units"],
        learning_rate=config["learning_rate"],
    )

    history = train_model(
        model,
        dataset.X[train_mask],
        dataset.y[train_mask],
        dataset.X[validation_mask],
        dataset.y[validation_mask],
        epochs=config["maximum_epochs"],
        batch_size=config["batch_size"],
        patience=config["early_stopping_patience"],
        verbose=1,
    )

    save_model(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    BEST_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    test_predictions = predict_sequence_subset(
        model, dataset, test_mask, frame, scaler, SEASONAL_PERIOD
    )
    test_metrics = calculate_metrics(
        test_predictions["Actual"], test_predictions["Predicted"]
    )
    test_indices = dataset.target_indices[test_mask]
    baselines = baseline_predictions(
        frame, test_indices, fit_end=validation_end, seasonal_period=SEASONAL_PERIOD
    )
    comparison = comparison_table(
        test_predictions["Actual"].to_numpy(),
        test_predictions["Predicted"].to_numpy(),
        baselines,
    )

    future_forecast = recursive_forecast(
        frame, model, scaler, horizon=24, lookback=LOOKBACK, seasonal_period=SEASONAL_PERIOD
    )
    forecast_summary = summarize_forecast(future_forecast)

    history_frame = pd.DataFrame(history.history)
    history_frame.index.name = "Epoch"
    history_frame.index = history_frame.index + 1

    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)
    future_forecast.to_csv(OUTPUT_DIR / "future_forecast_24_months.csv", index=False)
    history_frame.to_csv(OUTPUT_DIR / "training_history.csv")

    metrics_payload = {
        "test_metrics": test_metrics.to_dict(),
        "baseline_comparison": comparison.to_dict(orient="records"),
        "forecast_summary": forecast_summary,
    }
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    metadata = {
        "project": "Airline Passenger Forecasting using a Seasonally Adjusted LSTM",
        "model_version": "2.0.0",
        "backend": os.environ.get("KERAS_BACKEND", "jax"),
        "date_column": "Month",
        "target_column": "Passengers",
        "target_unit": "thousands of passengers",
        "transform": "year-over-year log growth",
        "feature_names": list(dataset.feature_names),
        "lookback": LOOKBACK,
        "seasonal_period": SEASONAL_PERIOD,
        "effective_raw_history_months": LOOKBACK + SEASONAL_PERIOD,
        "scaler": "StandardScaler fitted only on training seasonal-growth values",
        "data_start": frame["Month"].min().strftime("%Y-%m-%d"),
        "data_end": frame["Month"].max().strftime("%Y-%m-%d"),
        "training_target_end": frame.iloc[train_end - 1]["Month"].strftime("%Y-%m-%d"),
        "validation_start": frame.iloc[train_end]["Month"].strftime("%Y-%m-%d"),
        "validation_end": frame.iloc[validation_end - 1]["Month"].strftime("%Y-%m-%d"),
        "test_start": frame.iloc[validation_end]["Month"].strftime("%Y-%m-%d"),
        "test_end": frame.iloc[-1]["Month"].strftime("%Y-%m-%d"),
        "training_sequence_count": int(train_mask.sum()),
        "validation_sequence_count": int(validation_mask.sum()),
        "test_sequence_count": int(test_mask.sum()),
        "preprocessing_notes": preprocessing_notes,
        "configuration": config,
        "test_metrics": test_metrics.to_dict(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda line: summary_buffer.write(line + "\n"))
    (OUTPUT_DIR / "model_summary.txt").write_text(summary_buffer.getvalue(), encoding="utf-8")

    plot_passenger_trend(frame, OUTPUT_DIR / "passenger_trend.png")
    plot_seasonal_pattern(frame, OUTPUT_DIR / "seasonal_pattern.png")
    plot_training_curve(history.history, OUTPUT_DIR / "training_curve.png")
    plot_actual_vs_predicted(test_predictions, OUTPUT_DIR / "actual_vs_predicted.png")
    plot_residuals(test_predictions, OUTPUT_DIR / "residual_plot.png")
    plot_forecast(frame, future_forecast, OUTPUT_DIR / "forecast_plot.png")
    plot_baseline_comparison(comparison, OUTPUT_DIR / "baseline_comparison.png")

    print("\nTraining complete.")
    print(json.dumps(test_metrics.to_dict(), indent=2))
    print(f"Artifacts saved under: {MODEL_DIR}")


if __name__ == "__main__":
    main()
