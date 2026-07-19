from __future__ import annotations

from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


class NumpyBitcoinLSTM:
    """Backend-free inference for the packaged two-layer Keras LSTM."""

    REQUIRED_KEYS = {
        "lstm1_kernel",
        "lstm1_recurrent_kernel",
        "lstm1_bias",
        "lstm2_kernel",
        "lstm2_recurrent_kernel",
        "lstm2_bias",
        "dense1_kernel",
        "dense1_bias",
        "dense2_kernel",
        "dense2_bias",
    }

    def __init__(self, weights_path: str | Path):
        loaded = np.load(weights_path)
        missing = self.REQUIRED_KEYS.difference(loaded.files)
        if missing:
            raise ValueError(f"Missing weight arrays: {sorted(missing)}")
        self.weights = {name: loaded[name].astype(np.float32) for name in loaded.files}

    @staticmethod
    def _lstm_layer(
        x: np.ndarray,
        kernel: np.ndarray,
        recurrent_kernel: np.ndarray,
        bias: np.ndarray,
        return_sequences: bool,
    ) -> np.ndarray:
        batch_size, timesteps, _ = x.shape
        units = recurrent_kernel.shape[0]
        h = np.zeros((batch_size, units), dtype=np.float32)
        c = np.zeros((batch_size, units), dtype=np.float32)
        outputs: list[np.ndarray] = []

        for timestep in range(timesteps):
            combined = x[:, timestep, :] @ kernel + h @ recurrent_kernel + bias
            input_gate, forget_gate, candidate, output_gate = np.split(combined, 4, axis=1)
            input_gate = _sigmoid(input_gate)
            forget_gate = _sigmoid(forget_gate)
            candidate = np.tanh(candidate)
            output_gate = _sigmoid(output_gate)
            c = forget_gate * c + input_gate * candidate
            h = output_gate * np.tanh(c)
            outputs.append(h)

        if return_sequences:
            return np.stack(outputs, axis=1)
        return h

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float32)
        if values.ndim == 2:
            values = values[np.newaxis, ...]
        if values.ndim != 3:
            raise ValueError("Input must have shape (batch, timesteps, features).")

        w = self.weights
        values = self._lstm_layer(
            values,
            w["lstm1_kernel"],
            w["lstm1_recurrent_kernel"],
            w["lstm1_bias"],
            return_sequences=True,
        )
        values = self._lstm_layer(
            values,
            w["lstm2_kernel"],
            w["lstm2_recurrent_kernel"],
            w["lstm2_bias"],
            return_sequences=False,
        )
        values = np.maximum(0.0, values @ w["dense1_kernel"] + w["dense1_bias"])
        return values @ w["dense2_kernel"] + w["dense2_bias"]
