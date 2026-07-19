"""Utilities for exposing attention scores from the saved inference decoder."""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras


def build_attention_scoring_decoder(
    decoder_model: keras.Model,
    max_source_length: int,
    latent_dimension: int,
) -> keras.Model:
    """Rebuild the decoder graph with additive-attention scores as an output.

    The uploaded inference decoder returns token probabilities and LSTM states.
    This function reuses its trained layers and additionally requests the
    alignment scores supported by Keras ``AdditiveAttention``.
    """

    token_input = keras.Input(shape=(1,), dtype="int32", name="attention_token_input")
    encoder_outputs = keras.Input(
        shape=(max_source_length, latent_dimension), name="attention_encoder_outputs"
    )
    state_h = keras.Input(shape=(latent_dimension,), name="attention_state_h")
    state_c = keras.Input(shape=(latent_dimension,), name="attention_state_c")

    embedded = decoder_model.get_layer("decoder_embedding")(token_input)
    decoder_output, next_h, next_c = decoder_model.get_layer("decoder_lstm")(
        embedded, initial_state=[state_h, state_c]
    )
    context, scores = decoder_model.get_layer("attention_layer")(
        [decoder_output, encoder_outputs], return_attention_scores=True
    )
    combined = decoder_model.get_layer("concatenate")([decoder_output, context])
    probabilities = decoder_model.get_layer("time_distributed_output")(combined)

    return keras.Model(
        [token_input, encoder_outputs, state_h, state_c],
        [probabilities, next_h, next_c, scores],
        name="attention_scoring_decoder",
    )
