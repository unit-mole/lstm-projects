from __future__ import annotations

from pathlib import Path

import numpy as np

from .sequence_generation import ensure_model_shape


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _lstm_forward(
    inputs: np.ndarray,
    kernel: np.ndarray,
    recurrent_kernel: np.ndarray,
    bias: np.ndarray,
    return_sequences: bool,
) -> np.ndarray:
    values = np.asarray(inputs, dtype="float32")
    batch_size, timesteps, _ = values.shape
    units = recurrent_kernel.shape[0]
    hidden = np.zeros((batch_size, units), dtype="float32")
    cell = np.zeros((batch_size, units), dtype="float32")
    outputs: list[np.ndarray] = []

    for timestep in range(timesteps):
        combined = (
            values[:, timestep, :] @ kernel
            + hidden @ recurrent_kernel
            + bias
        )
        input_gate, forget_gate, candidate, output_gate = np.split(
            combined,
            4,
            axis=-1,
        )
        input_gate = _sigmoid(input_gate)
        forget_gate = _sigmoid(forget_gate)
        candidate = np.tanh(candidate)
        output_gate = _sigmoid(output_gate)

        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * np.tanh(cell)
        outputs.append(hidden.copy())

    sequence = np.stack(outputs, axis=1)
    return sequence if return_sequences else hidden


class NumpyECGAutoencoder:
    """Backend-free inference matching the supplied Keras autoencoder."""

    def __init__(self, weights_path: str | Path) -> None:
        arrays = np.load(weights_path)
        for key in arrays.files:
            setattr(self, key, arrays[key].astype("float32"))

    def reconstruct(self, signals: np.ndarray) -> np.ndarray:
        values = ensure_model_shape(signals)
        encoded_sequence = _lstm_forward(
            values,
            self.encoder_lstm_1_kernel,
            self.encoder_lstm_1_recurrent,
            self.encoder_lstm_1_bias,
            return_sequences=True,
        )
        latent = _lstm_forward(
            encoded_sequence,
            self.encoder_lstm_2_kernel,
            self.encoder_lstm_2_recurrent,
            self.encoder_lstm_2_bias,
            return_sequences=False,
        )
        repeated = np.repeat(latent[:, None, :], values.shape[1], axis=1)
        decoded = _lstm_forward(
            repeated,
            self.decoder_lstm_1_kernel,
            self.decoder_lstm_1_recurrent,
            self.decoder_lstm_1_bias,
            return_sequences=True,
        )
        decoded = _lstm_forward(
            decoded,
            self.decoder_lstm_2_kernel,
            self.decoder_lstm_2_recurrent,
            self.decoder_lstm_2_bias,
            return_sequences=True,
        )
        return decoded @ self.output_kernel + self.output_bias

    def reconstruct_in_batches(
        self,
        signals: np.ndarray,
        batch_size: int = 128,
    ) -> np.ndarray:
        values = ensure_model_shape(signals)
        return np.concatenate(
            [
                self.reconstruct(values[index : index + batch_size])
                for index in range(0, len(values), batch_size)
            ],
            axis=0,
        )
