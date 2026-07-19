from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SAMPLE_DATA_PATH, THRESHOLD, WEIGHTS_PATH
from src.data_preprocessing import frame_to_sequences, prepare_ecg_frame
from src.inference_pipeline import ECGInferenceService


def build_service() -> ECGInferenceService:
    return ECGInferenceService.from_artifacts(
        weights_path=WEIGHTS_PATH,
        threshold=THRESHOLD,
    )


def test_normal_example_scores_below_threshold() -> None:
    frame = prepare_ecg_frame(pd.read_csv(SAMPLE_DATA_PATH))
    sequences = frame_to_sequences(frame)
    result = build_service().analyze_signal(sequences[0])

    assert result.reconstruction.shape == (140,)
    assert result.reconstruction_error < THRESHOLD
    assert result.predicted_label == 0


def test_anomaly_example_scores_above_threshold() -> None:
    frame = prepare_ecg_frame(pd.read_csv(SAMPLE_DATA_PATH))
    anomaly_index = int(frame.index[frame["label"] == 1][0])
    sequences = frame_to_sequences(frame)
    result = build_service().analyze_signal(sequences[anomaly_index])

    assert result.reconstruction_error >= THRESHOLD
    assert result.predicted_label == 1
    assert np.isclose(result.temporal_focus.sum(), 1.0)


def test_batch_scoring_has_valid_outputs() -> None:
    frame = prepare_ecg_frame(pd.read_csv(SAMPLE_DATA_PATH)).head(20)
    scored = build_service().score_frame(frame)

    assert len(scored) == 20
    assert np.isfinite(scored["reconstruction_error"]).all()
    assert np.isfinite(scored["anomaly_score"]).all()
    assert set(scored["predicted_label"].unique()).issubset({0, 1})
