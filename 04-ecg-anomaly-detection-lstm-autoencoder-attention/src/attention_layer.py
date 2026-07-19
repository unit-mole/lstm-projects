from __future__ import annotations

import numpy as np


ATTENTION_QUALIFICATION = (
    "The supplied deployed model has no trainable attention layer. "
    "temporal_focus_weights() is a post-hoc explainability method based on pointwise "
    "reconstruction error. build_attention_pooling() is used only by optional retraining."
)


def temporal_focus_weights(
    original: np.ndarray,
    reconstructed: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert pointwise reconstruction errors into normalized temporal focus weights."""
    errors = np.abs(
        np.asarray(original, dtype=float).reshape(-1)
        - np.asarray(reconstructed, dtype=float).reshape(-1)
    )
    scale = max(float(temperature), 1e-6)
    logits = errors / scale
    logits = logits - logits.max()
    weights = np.exp(logits)
    denominator = weights.sum()
    return weights / denominator if denominator > 0 else np.zeros_like(weights)


def build_attention_pooling(sequence_length: int):
    """Return a trainable Keras temporal-attention pooling branch for retraining."""
    try:
        import keras
    except ImportError as exc:
        raise ImportError(
            "Install the full training requirements before building the attention model."
        ) from exc

    score_layer = keras.layers.Dense(1, activation="tanh", name="attention_score")
    softmax_layer = keras.layers.Softmax(axis=1, name="attention_weights")
    multiply_layer = keras.layers.Multiply(name="attention_multiply")
    average_layer = keras.layers.GlobalAveragePooling1D(name="attention_average")
    rescale_layer = keras.layers.Rescaling(sequence_length, name="attention_sum")

    def apply(encoded_sequence):
        scores = score_layer(encoded_sequence)
        weights = softmax_layer(scores)
        weighted_sequence = multiply_layer([encoded_sequence, weights])
        context = average_layer(weighted_sequence)
        return rescale_layer(context), weights

    return apply
