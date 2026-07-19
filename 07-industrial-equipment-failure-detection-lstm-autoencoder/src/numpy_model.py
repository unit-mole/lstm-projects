from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


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
    outputs: list[np.ndarray] = []

    # Keras gate order is input, forget, candidate/cell, output.
    for step in range(time_steps):
        gates = inputs[:, step, :] @ kernel + hidden @ recurrent_kernel + bias
        input_gate, forget_gate, candidate_gate, output_gate = np.split(gates, 4, axis=1)
        input_gate = _sigmoid(input_gate)
        forget_gate = _sigmoid(forget_gate)
        candidate_gate = np.tanh(candidate_gate)
        output_gate = _sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * candidate_gate
        hidden = output_gate * np.tanh(cell)
        outputs.append(hidden)

    return np.stack(outputs, axis=1) if return_sequences else hidden


class NumpyLSTMAutoencoder:
    """Portable inference-only reader for this project's Keras v3 model artifact.

    It reproduces the four LSTM layers and TimeDistributed dense output directly
    from the weights stored inside the `.keras` archive. This keeps the public
    Streamlit demo lightweight while preserving the trained LSTM model behavior.
    """

    def __init__(
        self,
        lstm_weights: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        dense_kernel: np.ndarray,
        dense_bias: np.ndarray,
        sequence_length: int,
        n_features: int,
    ) -> None:
        self.lstm_weights = lstm_weights
        self.dense_kernel = dense_kernel
        self.dense_bias = dense_bias
        self.sequence_length = sequence_length
        self.n_features = n_features

    @classmethod
    def from_keras(cls, model_path: str | Path) -> "NumpyLSTMAutoencoder":
        model_path = Path(model_path)
        with zipfile.ZipFile(model_path) as archive:
            config = json.loads(archive.read("config.json"))
            weights_bytes = archive.read("model.weights.h5")

        input_shape = None
        for layer in config["config"]["layers"]:
            if layer.get("class_name") == "InputLayer":
                input_shape = layer["config"].get("batch_shape") or layer["config"].get("batch_input_shape")
                break
        if not input_shape or len(input_shape) != 3:
            raise ValueError("Unable to determine model input shape from Keras config.")
        sequence_length, n_features = int(input_shape[1]), int(input_shape[2])

        with h5py.File(io.BytesIO(weights_bytes), "r") as h5:
            lstm_weights = []
            for layer_name in ["lstm", "lstm_1", "lstm_2", "lstm_3"]:
                path = f"layers/{layer_name}/cell/vars"
                lstm_weights.append(tuple(np.array(h5[f"{path}/{index}"]) for index in range(3)))
            dense_kernel = np.array(h5["layers/time_distributed/layer/vars/0"])
            dense_bias = np.array(h5["layers/time_distributed/layer/vars/1"])

        return cls(lstm_weights, dense_kernel, dense_bias, sequence_length, n_features)

    def predict(self, sequences: np.ndarray, batch_size: int = 256, verbose: int = 0) -> np.ndarray:
        del verbose
        inputs = np.asarray(sequences, dtype=np.float32)
        expected = (self.sequence_length, self.n_features)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
            raise ValueError(f"Expected input shape [samples, {expected[0]}, {expected[1]}].")

        outputs = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            encoded = _lstm_forward(batch, *self.lstm_weights[0], return_sequences=True)
            encoded = _lstm_forward(encoded, *self.lstm_weights[1], return_sequences=False)
            decoded = np.repeat(encoded[:, None, :], self.sequence_length, axis=1)
            decoded = _lstm_forward(decoded, *self.lstm_weights[2], return_sequences=True)
            decoded = _lstm_forward(decoded, *self.lstm_weights[3], return_sequences=True)
            outputs.append(decoded @ self.dense_kernel + self.dense_bias)
        return np.concatenate(outputs, axis=0)
