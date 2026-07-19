from __future__ import annotations

import argparse

from src.model_training import train_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the Bitcoin LSTM project.")
    parser.add_argument("--csv", default=None, help="Optional local OHLCV CSV path.")
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--period", default="8y")
    parser.add_argument("--look-back", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_project(
        csv_path=args.csv,
        ticker=args.ticker,
        period=args.period,
        look_back=args.look_back,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("Strict held-out test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
