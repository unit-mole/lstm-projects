from pathlib import Path

import numpy as np

from src.synthetic_weather import generate_weather_sequences


def main() -> None:
    X, future_y = generate_weather_sequences(n_samples=24, future_frames=6, seed=42)
    destination = Path(__file__).resolve().parents[1] / "data" / "sample_weather_sequences.npz"
    np.savez_compressed(destination, X=X, y=future_y[:, 0], future_y=future_y)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
