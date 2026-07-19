from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.weather_preprocessing import validate_sequence_shape


def load_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model(path: str | Path):
    import tensorflow as tf
    return tf.keras.models.load_model(Path(path), compile=False)


def predict_next_frame(model, sequence: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    sequence = validate_sequence_shape(
        sequence,
        metadata["input_frames"],
        metadata["height"],
        metadata["width"],
        metadata["channels"],
    )
    prediction = model.predict(sequence[None, ...], verbose=0)[0]
    return np.clip(np.asarray(prediction, dtype="float32"), 0.0, 1.0)
