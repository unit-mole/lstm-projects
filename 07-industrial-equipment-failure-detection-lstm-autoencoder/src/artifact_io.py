from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .numpy_model import NumpyLSTMAutoencoder


class PortableStandardScaler:
    """Minimal StandardScaler transform reconstructed from JSON parameters."""

    def __init__(self, mean: list[float], scale: list[float], feature_names: list[str]):
        self.mean_ = np.asarray(mean, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = len(feature_names)

    def transform(self, values):
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, received {array.shape[-1]}.")
        return (array - self.mean_) / self.scale_

    def inverse_transform(self, values):
        array = np.asarray(values, dtype=float)
        return array * self.scale_ + self.mean_


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_scaler(model_dir: str | Path):
    """Load portable JSON parameters first; retain the pickle for training interoperability."""
    model_dir = Path(model_dir)
    json_path = model_dir / "scaler.json"
    if json_path.exists():
        payload = load_json(json_path)
        return PortableStandardScaler(
            mean=payload["mean"], scale=payload["scale"], feature_names=payload["feature_names"]
        )
    return joblib.load(model_dir / "scaler.pkl")


def load_model(model_path: str | Path, prefer_tensorflow: bool = False):
    """Load TensorFlow when requested and available; otherwise use portable NumPy."""
    model_path = Path(model_path)
    if prefer_tensorflow:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path, compile=False), "TensorFlow/Keras"
        except Exception:
            pass
    return NumpyLSTMAutoencoder.from_keras(model_path), "Portable NumPy/Keras-weights"
