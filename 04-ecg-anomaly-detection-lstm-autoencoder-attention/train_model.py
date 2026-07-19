from __future__ import annotations

import argparse
from pathlib import Path

from src.model_training import train_attention_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain a temporal-attention LSTM Autoencoder on synthetic ECG-like data."
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_attention_model(
        output_dir=Path.cwd(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
