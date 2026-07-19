from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly_detection import DetectionResult, classify_reconstruction, reconstruction_errors
from .cloud_inference import NumpyECGAutoencoder
from .data_preprocessing import frame_to_sequences
from .sequence_generation import ensure_model_shape


@dataclass
class ECGInferenceService:
    model: NumpyECGAutoencoder
    threshold: float

    @classmethod
    def from_artifacts(
        cls,
        weights_path: str | Path,
        threshold: float,
    ) -> "ECGInferenceService":
        return cls(
            model=NumpyECGAutoencoder(weights_path),
            threshold=float(threshold),
        )

    def analyze_signal(self, signal: np.ndarray) -> DetectionResult:
        values = ensure_model_shape(signal)
        reconstruction = self.model.reconstruct(values)[0]
        return classify_reconstruction(
            values[0],
            reconstruction,
            self.threshold,
        )

    def score_sequences(
        self,
        signals: np.ndarray,
        batch_size: int = 128,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = ensure_model_shape(signals)
        reconstruction = self.model.reconstruct_in_batches(
            values,
            batch_size=batch_size,
        )
        errors = reconstruction_errors(values, reconstruction)
        labels = (errors >= self.threshold).astype(int)
        return reconstruction, errors, labels

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        sequences = frame_to_sequences(frame)
        _, errors, labels = self.score_sequences(sequences)
        scored = frame[["signal_id", "label", "anomaly_type"]].copy()
        scored["predicted_label"] = labels
        scored["predicted_status"] = np.where(
            labels == 1,
            "Anomalous pattern",
            "Normal pattern",
        )
        scored["reconstruction_error"] = errors
        scored["threshold"] = self.threshold
        scored["anomaly_score"] = errors / self.threshold
        return scored
