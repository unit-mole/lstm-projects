from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DATA_DIR
from src.forecasting_pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multivariate Stacked LSTM forecasting model.")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "hourly_energy.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run_training_pipeline(
        data_path=arguments.data,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
    )
    print("Training complete.")
    print(result)
