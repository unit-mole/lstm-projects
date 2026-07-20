"""Backend-safe utilities for obtaining decoder attention scores at inference."""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np


def decoder_step_with_attention(
    decoder_model: keras.Model,
    token_input: np.ndarray,
    encoder_outputs: np.ndarray,
    state_h: np.ndarray,
    state_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one decoder step and return probabilities, states, and attention.

    The saved decoder already contains the trained embedding, LSTM, additive
    attention, concatenation, and output layers. Calling those layers directly
    with runtime tensors avoids constructing a second Functional model around
    layers from a deserialized model. That graph reconstruction can fail on
    some Keras/backend combinations in hosted environments.
    """

    token_tensor = keras.ops.convert_to_tensor(token_input, dtype="int32")
    encoder_tensor = keras.ops.convert_to_tensor(encoder_outputs)
    state_h_tensor = keras.ops.convert_to_tensor(state_h)
    state_c_tensor = keras.ops.convert_to_tensor(state_c)

    embedding_layer = decoder_model.get_layer("decoder_embedding")
    lstm_layer = decoder_model.get_layer("decoder_lstm")
    attention_layer = decoder_model.get_layer("attention_layer")
    concatenate_layer = decoder_model.get_layer("concatenate")
    output_layer = decoder_model.get_layer("time_distributed_output")

    embedded = embedding_layer(token_tensor)
    decoder_output, next_h, next_c = lstm_layer(
        embedded,
        initial_state=[state_h_tensor, state_c_tensor],
        training=False,
    )
    context, attention_scores = attention_layer(
        [decoder_output, encoder_tensor],
        return_attention_scores=True,
        training=False,
    )
    combined = concatenate_layer([decoder_output, context])
    probabilities = output_layer(combined, training=False)

    return (
        np.asarray(probabilities),
        np.asarray(next_h),
        np.asarray(next_c),
        np.asarray(attention_scores),
    )
