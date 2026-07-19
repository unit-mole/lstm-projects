"""Command-line retraining entrypoint for the synthetic moving-object experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.model_training import train_model
from src.sequence_generation import generate_moving_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ConvLSTM next-frame predictor.")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("models/convlstm_video_prediction_retrained.keras")
    )
    args = parser.parse_args()

    x_all, y_all = generate_moving_sequences(n_samples=args.samples, seed=args.seed)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x_all, y_all, test_size=0.30, random_state=args.seed
    )
    x_val, _, y_val, _ = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=args.seed
    )
    _, history = train_model(
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=args.output,
        seed=args.seed,
    )
    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved model: {args.output}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    main()
