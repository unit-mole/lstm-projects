"""Regenerate sample visualizations after the model or safe sample data changes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.config import MODEL_PATH, OUTPUT_DIR, SAMPLE_DATA_PATH
from src.data_preprocessing import load_sample_sequences
from src.prediction_pipeline import load_prediction_model, recursive_predict
from src.visualization import plot_input_sequence, plot_prediction_comparison


def main() -> None:
    samples = load_sample_sequences(SAMPLE_DATA_PATH)
    sequence = samples["X"][0]
    actual = samples["y"][0]
    model = load_prediction_model(MODEL_PATH)
    prediction = model.predict(sequence[None, ...], verbose=0)[0]
    future = recursive_predict(model, sequence, future_steps=6)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure = plot_input_sequence(sequence)
    figure.savefig(OUTPUT_DIR / "sample_input_frames_regenerated.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure = plot_prediction_comparison(sequence, prediction, actual)
    figure.savefig(OUTPUT_DIR / "actual_vs_predicted_regenerated.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    np.save(OUTPUT_DIR / "recursive_future_frames.npy", future)
    print("Demo assets regenerated.")


if __name__ == "__main__":
    main()
