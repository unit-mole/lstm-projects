from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))

def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)

def _lstm_forward(inputs: np.ndarray, kernel: np.ndarray, recurrent_kernel: np.ndarray,
                  bias: np.ndarray, initial_h: np.ndarray | None = None,
                  initial_c: np.ndarray | None = None, mask: np.ndarray | None = None):
    values = np.asarray(inputs, dtype=np.float32)
    batch_size, timesteps, _ = values.shape
    units = recurrent_kernel.shape[0]
    h = np.zeros((batch_size, units), dtype=np.float32) if initial_h is None else np.asarray(initial_h, dtype=np.float32).copy()
    c = np.zeros((batch_size, units), dtype=np.float32) if initial_c is None else np.asarray(initial_c, dtype=np.float32).copy()
    outputs = []
    for timestep in range(timesteps):
        combined = values[:, timestep, :] @ kernel + h @ recurrent_kernel + bias
        input_gate, forget_gate, candidate, output_gate = np.split(combined, 4, axis=-1)
        input_gate = _sigmoid(input_gate)
        forget_gate = _sigmoid(forget_gate)
        candidate = np.tanh(candidate)
        output_gate = _sigmoid(output_gate)
        updated_c = forget_gate * c + input_gate * candidate
        updated_h = output_gate * np.tanh(updated_c)
        if mask is not None:
            active = mask[:, timestep:timestep + 1].astype(np.float32)
            h = active * updated_h + (1.0 - active) * h
            c = active * updated_c + (1.0 - active) * c
        else:
            h, c = updated_h, updated_c
        outputs.append(h.copy())
    return np.stack(outputs, axis=1), h, c

@dataclass
class GenerationOutput:
    response: str
    generated_tokens: list[str]
    token_confidences: list[float]
    average_confidence: float
    attention_weights: np.ndarray
    model_input_ids: np.ndarray

class NumpySeq2SeqAttention:
    """Backend-free inference matching the supplied Keras Seq2Seq model."""

    def __init__(self, weights_path: str | Path) -> None:
        arrays = np.load(weights_path)
        for key in arrays.files:
            setattr(self, key, arrays[key].astype(np.float32))

    def encode(self, source_ids: np.ndarray):
        ids = np.asarray(source_ids, dtype=np.int32)
        embeddings = self.encoder_embedding[ids]
        return _lstm_forward(
            embeddings, self.encoder_kernel, self.encoder_recurrent_kernel,
            self.encoder_bias, mask=ids != 0
        )

    def decoder_step(self, token_ids: np.ndarray, encoder_outputs: np.ndarray,
                     state_h: np.ndarray, state_c: np.ndarray):
        ids = np.asarray(token_ids, dtype=np.int32)
        embeddings = self.decoder_embedding[ids]
        decoder_outputs, updated_h, updated_c = _lstm_forward(
            embeddings, self.decoder_kernel, self.decoder_recurrent_kernel,
            self.decoder_bias, initial_h=state_h, initial_c=state_c
        )
        query = decoder_outputs[:, :, None, :]
        keys = encoder_outputs[:, None, :, :]
        scores = np.sum(self.attention_scale * np.tanh(query + keys), axis=-1)
        attention = _softmax(scores, axis=-1)
        context = attention @ encoder_outputs
        combined = np.concatenate([decoder_outputs, context], axis=-1)
        probabilities = _softmax(combined @ self.output_kernel + self.output_bias, axis=-1)
        return probabilities, updated_h, updated_c, attention

    def generate(self, source_ids: np.ndarray, target_index_word: dict[int, str],
                 start_token_id: int, end_token_id: int, max_target_length: int):
        encoder_outputs, state_h, state_c = self.encode(source_ids)
        current_token = int(start_token_id)
        generated_tokens, token_confidences, attention_rows = [], [], []
        for _ in range(max_target_length - 1):
            probabilities, state_h, state_c, attention = self.decoder_step(
                np.asarray([[current_token]], dtype=np.int32),
                encoder_outputs, state_h, state_c
            )
            distribution = probabilities[0, -1, :]
            sampled_id = int(np.argmax(distribution))
            confidence = float(distribution[sampled_id])
            if sampled_id in {0, int(end_token_id)}:
                break
            sampled_word = target_index_word.get(sampled_id, "")
            if sampled_word and sampled_word not in {"sostok", "eostok", "<unk>"}:
                generated_tokens.append(sampled_word)
                token_confidences.append(confidence)
                attention_rows.append(attention[0, 0, :].copy())
            current_token = sampled_id
        attention_matrix = np.stack(attention_rows, axis=0) if attention_rows else np.empty((0, source_ids.shape[1]), dtype=np.float32)
        average_confidence = float(np.mean(token_confidences)) if token_confidences else 0.0
        return GenerationOutput(
            response=" ".join(generated_tokens),
            generated_tokens=generated_tokens,
            token_confidences=token_confidences,
            average_confidence=average_confidence,
            attention_weights=attention_matrix,
            model_input_ids=np.asarray(source_ids, dtype=np.int32),
        )
