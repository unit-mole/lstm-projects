"""Reproduce the test metrics and baseline comparison for the saved model."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config import MODEL_PATH, OUTPUT_DIR
from src.model_evaluation import (
    build_comparison_table,
    frame_average_baseline,
    persistence_baseline,
)
from src.prediction_pipeline import batch_predict, load_prediction_model
from src.sequence_generation import generate_moving_sequences


def main() -> None:
    x_all, y_all = generate_moving_sequences(seed=42)
    _, x_temp, _, y_temp = train_test_split(x_all, y_all, test_size=0.30, random_state=42)
    _, x_test, _, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)
    model = load_prediction_model(MODEL_PATH)
    predictions = batch_predict(model, x_test)
    table = build_comparison_table(
        y_test,
        {
            "Persistence (last frame)": persistence_baseline(x_test),
            "Frame average": frame_average_baseline(x_test),
            "ConvLSTM": predictions,
        },
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "baseline_metrics_reproduced.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
