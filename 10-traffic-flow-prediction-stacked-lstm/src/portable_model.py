"""Lightweight NumPy inference for the supplied Keras Stacked LSTM artifact."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _lstm_forward(
    inputs: np.ndarray,
    kernel: np.ndarray,
    recurrent_kernel: np.ndarray,
    bias: np.ndarray,
    return_sequences: bool,
) -> np.ndarray:
    batch_size, time_steps, _ = inputs.shape
    units = recurrent_kernel.shape[0]
    hidden = np.zeros((batch_size, units), dtype=np.float32)
    cell = np.zeros((batch_size, units), dtype=np.float32)
    outputs = []

    for step in range(time_steps):
        projected = (
            inputs[:, step, :] @ kernel
            + hidden @ recurrent_kernel
            + bias
        )
        input_gate, forget_gate, candidate, output_gate = np.split(
            projected,
            4,
            axis=1,
        )
        input_gate = _sigmoid(input_gate)
        forget_gate = _sigmoid(forget_gate)
        candidate = np.tanh(candidate)
        output_gate = _sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * np.tanh(cell)
        outputs.append(hidden)

    if return_sequences:
        return np.stack(outputs, axis=1)
    return hidden


class PortableStackedLSTM:
    """Read the Keras zip artifact and execute inference without TensorFlow."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.config, self.weights = self._load_artifact(self.model_path)

    @staticmethod
    def _load_artifact(
        model_path: Path,
    ) -> tuple[dict, dict[str, list[np.ndarray]]]:
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        with zipfile.ZipFile(model_path, "r") as archive:
            config = json.loads(archive.read("config.json"))
            weight_bytes = archive.read("model.weights.h5")

        weights: dict[str, list[np.ndarray]] = {}
        with h5py.File(io.BytesIO(weight_bytes), "r") as weight_file:
            for layer_name in ["lstm", "lstm_1", "lstm_2"]:
                base = f"layers/{layer_name}/cell/vars"
                weights[layer_name] = [
                    weight_file[f"{base}/0"][()],
                    weight_file[f"{base}/1"][()],
                    weight_file[f"{base}/2"][()],
                ]
            for layer_name in ["dense", "dense_1"]:
                base = f"layers/{layer_name}/vars"
                weights[layer_name] = [
                    weight_file[f"{base}/0"][()],
                    weight_file[f"{base}/1"][()],
                ]
        return config, weights

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """Return scaled one-step congestion predictions."""
        values = np.asarray(sequences, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                "Expected a 3D array: samples × time steps × features."
            )

        values = _lstm_forward(
            values,
            *self.weights["lstm"],
            return_sequences=True,
        )
        values = _lstm_forward(
            values,
            *self.weights["lstm_1"],
            return_sequences=True,
        )
        values = _lstm_forward(
            values,
            *self.weights["lstm_2"],
            return_sequences=False,
        )
        values = (
            values @ self.weights["dense"][0]
            + self.weights["dense"][1]
        )
        values = np.maximum(values, 0.0)
        values = (
            values @ self.weights["dense_1"][0]
            + self.weights["dense_1"][1]
        )
        return values.reshape(-1)
