from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_DATA_PATH, THRESHOLD, WEIGHTS_PATH
from src.data_preprocessing import frame_to_sequences, prepare_ecg_frame
from src.inference_pipeline import ECGInferenceService


REQUIRED_FILES = [
    "README.md",
    "README_HOSTING.md",
    "app/streamlit_app.py",
    "app/requirements.txt",
    "data/sample_ecg_signals.csv",
    "data/README_data.md",
    "models/lstm_autoencoder_ecg.keras",
    "models/lstm_autoencoder_ecg_weights.npz",
    "models/ecg_meta_supplied.json",
    "models/model_metadata.json",
    "notebooks/ecg_anomaly_detection_lstm_autoencoder_attention.ipynb",
    "outputs/model_metrics.json",
    "outputs/test_predictions.csv",
    "outputs/training_curve.png",
    "outputs/confusion_matrix.png",
    "src/cloud_inference.py",
    "src/inference_pipeline.py",
    "tests/test_inference.py",
]


def main() -> None:
    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (PROJECT_ROOT / relative).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Required project files are missing: {missing}")

    frame = prepare_ecg_frame(pd.read_csv(SAMPLE_DATA_PATH))
    sequences = frame_to_sequences(frame)
    service = ECGInferenceService.from_artifacts(
        WEIGHTS_PATH,
        threshold=THRESHOLD,
    )
    scored = service.score_frame(frame.head(40))
    normal_result = service.analyze_signal(sequences[0])
    anomaly_index = int(frame.index[frame["label"] == 1][0])
    anomaly_result = service.analyze_signal(sequences[anomaly_index])

    checks = {
        "required_files": "passed",
        "sample_rows": int(len(frame)),
        "sequence_shape": list(sequences.shape),
        "normal_error": float(normal_result.reconstruction_error),
        "normal_below_threshold": bool(
            normal_result.reconstruction_error < THRESHOLD
        ),
        "anomaly_error": float(anomaly_result.reconstruction_error),
        "anomaly_above_threshold": bool(
            anomaly_result.reconstruction_error >= THRESHOLD
        ),
        "batch_scores_finite": bool(
            np.isfinite(scored["reconstruction_error"]).all()
        ),
        "temporal_focus_normalized": bool(
            np.isclose(anomaly_result.temporal_focus.sum(), 1.0)
        ),
        "cloud_runtime": "NumPy LSTM Autoencoder inference",
    }

    if not all(
        [
            checks["normal_below_threshold"],
            checks["anomaly_above_threshold"],
            checks["batch_scores_finite"],
            checks["temporal_focus_normalized"],
        ]
    ):
        raise RuntimeError(f"Project validation failed: {checks}")

    print(json.dumps(checks, indent=2))
    print("ECG anomaly-detection project validation passed.")


if __name__ == "__main__":
    main()
