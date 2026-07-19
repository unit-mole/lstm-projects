"""Portable scaling based on JSON statistics saved during training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalingArtifacts:
    """Feature and target scaling statistics used by the trained model."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float
    feature_columns: list[str]
    sequence_length: int

    @classmethod
    def from_json(cls, path: str | Path) -> "ScalingArtifacts":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float32),
            target_mean=float(payload["target_mean"][0]),
            target_scale=float(payload["target_scale"][0]),
            feature_columns=list(payload["feature_cols"]),
            sequence_length=int(payload["seq_len"]),
        )

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.feature_columns].to_numpy(dtype=np.float32)
        return (values - self.feature_mean) / self.feature_scale

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return (array - self.target_mean) / self.target_scale

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return array * self.target_scale + self.target_mean
