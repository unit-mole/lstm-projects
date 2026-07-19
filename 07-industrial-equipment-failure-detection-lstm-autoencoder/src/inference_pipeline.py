from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly_detection import (
    build_prediction_frame,
    sensor_reconstruction_error,
    sequence_reconstruction_error,
)
from .artifact_io import load_json, load_model, load_scaler
from .sensor_preprocessing import prepare_inference_frame
from .sequence_generation import SequenceBatch, build_sequences


@dataclass(frozen=True)
class InferenceResult:
    clean_frame: pd.DataFrame
    scaled_frame: pd.DataFrame
    batch: SequenceBatch
    reconstructions: np.ndarray
    predictions: pd.DataFrame
    sensor_errors: np.ndarray
    backend_name: str


class PredictiveMaintenancePipeline:
    def __init__(self, model, scaler, metadata: dict, backend_name: str):
        self.model = model
        self.scaler = scaler
        self.metadata = metadata
        self.backend_name = backend_name

    @classmethod
    def from_artifacts(
        cls,
        model_dir: str | Path,
        prefer_tensorflow: bool = False,
    ) -> "PredictiveMaintenancePipeline":
        model_dir = Path(model_dir)
        metadata = load_json(model_dir / "model_metadata.json")
        scaler = load_scaler(model_dir)
        model, backend_name = load_model(
            model_dir / metadata["model_file"], prefer_tensorflow=prefer_tensorflow
        )
        return cls(model, scaler, metadata, backend_name)

    def score_dataframe(
        self,
        frame: pd.DataFrame,
        selected_unit=None,
        step_size: int | None = None,
    ) -> InferenceResult:
        clean, scaled, schema = prepare_inference_frame(
            frame, self.scaler, self.metadata, selected_unit=selected_unit
        )
        batch = build_sequences(
            scaled,
            schema,
            window_size=int(self.metadata["seq_len"]),
            step_size=int(step_size or self.metadata.get("step_size", 1)),
        )
        reconstructions = self.model.predict(batch.sequences, verbose=0)
        errors = sequence_reconstruction_error(batch.sequences, reconstructions, metric="mae")
        predictions = build_prediction_frame(
            batch, errors, float(self.metadata["threshold"])
        )
        sensor_errors = sensor_reconstruction_error(batch.sequences, reconstructions)
        return InferenceResult(
            clean_frame=clean,
            scaled_frame=scaled,
            batch=batch,
            reconstructions=reconstructions,
            predictions=predictions,
            sensor_errors=sensor_errors,
            backend_name=self.backend_name,
        )
