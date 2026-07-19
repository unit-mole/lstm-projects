"""Reproducible training pipeline for the Seq2Seq attention summarizer."""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
from pathlib import Path

import keras
import numpy as np

from .config import END_TOKEN, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH, OOV_TOKEN, SEED, START_TOKEN
from .data_preprocessing import generate_summarization_dataset, prepare_dataset, split_dataset
from .sequence_generation import pad_sequences_post, prepare_teacher_forcing_targets
from .tokenizer_utils import PortableTokenizer, fit_portable_tokenizer


def build_seq2seq_models(
    source_vocab_size: int,
    target_vocab_size: int,
    max_source_length: int = MAX_SOURCE_LENGTH,
    max_target_length: int = MAX_TARGET_LENGTH,
    embedding_dimension: int = 128,
    latent_dimension: int = 128,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Build training, encoder-inference, and decoder-inference models."""

    encoder_inputs = keras.Input(shape=(max_source_length,), name="encoder_inputs")
    encoder_embedding = keras.layers.Embedding(
        source_vocab_size,
        embedding_dimension,
        mask_zero=True,
        name="encoder_embedding",
    )(encoder_inputs)
    encoder_lstm = keras.layers.LSTM(
        latent_dimension,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm",
    )
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)

    decoder_inputs = keras.Input(shape=(max_target_length - 1,), name="decoder_inputs")
    decoder_embedding_layer = keras.layers.Embedding(
        target_vocab_size,
        embedding_dimension,
        mask_zero=True,
        name="decoder_embedding",
    )
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm = keras.layers.LSTM(
        latent_dimension,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm",
    )
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding, initial_state=[state_h, state_c]
    )

    attention_layer = keras.layers.AdditiveAttention(name="attention_layer")
    attention_output = attention_layer([decoder_outputs, encoder_outputs])
    concatenate_layer = keras.layers.Concatenate(axis=-1, name="concatenate")
    decoder_context = concatenate_layer([decoder_outputs, attention_output])
    output_layer = keras.layers.TimeDistributed(
        keras.layers.Dense(target_vocab_size, activation="softmax"),
        name="time_distributed_output",
    )
    decoder_predictions = output_layer(decoder_context)

    training_model = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_predictions, name="seq2seq_attention"
    )
    training_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    encoder_model = keras.Model(
        encoder_inputs,
        [encoder_outputs, state_h, state_c],
        name="summarization_encoder",
    )

    inference_token = keras.Input(shape=(1,), name="inference_token")
    inference_encoder_outputs = keras.Input(
        shape=(max_source_length, latent_dimension), name="inference_encoder_outputs"
    )
    inference_h = keras.Input(shape=(latent_dimension,), name="inference_h")
    inference_c = keras.Input(shape=(latent_dimension,), name="inference_c")
    inference_embedding = decoder_embedding_layer(inference_token)
    inference_decoder_output, next_h, next_c = decoder_lstm(
        inference_embedding, initial_state=[inference_h, inference_c]
    )
    inference_attention = attention_layer(
        [inference_decoder_output, inference_encoder_outputs]
    )
    inference_context = concatenate_layer(
        [inference_decoder_output, inference_attention]
    )
    inference_probabilities = output_layer(inference_context)
    decoder_model = keras.Model(
        [inference_token, inference_encoder_outputs, inference_h, inference_c],
        [inference_probabilities, next_h, next_c],
        name="summarization_decoder",
    )
    return training_model, encoder_model, decoder_model


def prepare_training_data(sample_count: int, seed: int):
    frame = prepare_dataset(generate_summarization_dataset(sample_count, seed))
    train, validation, test = split_dataset(frame, seed)
    source_tokenizer = fit_portable_tokenizer(train["article_clean"], OOV_TOKEN)
    target_tokenizer = fit_portable_tokenizer(train["summary_seq"], OOV_TOKEN)

    def source_sequences(values):
        return pad_sequences_post(
            source_tokenizer.texts_to_sequences(values), MAX_SOURCE_LENGTH
        )

    def target_sequences(values):
        return pad_sequences_post(
            target_tokenizer.texts_to_sequences(values), MAX_TARGET_LENGTH
        )

    prepared = {}
    for name, split in [("train", train), ("validation", validation), ("test", test)]:
        encoder = source_sequences(split["article_clean"])
        target = target_sequences(split["summary_seq"])
        decoder_input, decoder_target = prepare_teacher_forcing_targets(target)
        prepared[name] = (encoder, decoder_input, decoder_target)
    return prepared, source_tokenizer, target_tokenizer


def train_and_save(
    output_dir: str | Path,
    sample_count: int = 2500,
    epochs: int = 12,
    batch_size: int = 64,
    seed: int = SEED,
) -> dict[str, object]:
    """Train and save all artifacts required for inference."""

    keras.utils.set_random_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared, source_tokenizer, target_tokenizer = prepare_training_data(
        sample_count, seed
    )
    model, encoder, decoder = build_seq2seq_models(
        source_tokenizer.vocab_size, target_tokenizer.vocab_size
    )
    x_train, decoder_train, target_train = prepared["train"]
    x_validation, decoder_validation, target_validation = prepared["validation"]
    history = model.fit(
        [x_train, decoder_train],
        target_train[..., np.newaxis],
        validation_data=(
            [x_validation, decoder_validation],
            target_validation[..., np.newaxis],
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
            ),
        ],
        verbose=1,
    )

    model.save(output_dir / "seq2seq_summarization.keras")
    encoder.save(output_dir / "encoder_summarization.keras")
    decoder.save(output_dir / "decoder_summarization.keras")
    source_tokenizer.save(output_dir / "source_tokenizer.json")
    target_tokenizer.save(output_dir / "target_tokenizer.json")
    payload = {
        "sample_count": sample_count,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "seed": seed,
        "source_vocab_size": source_tokenizer.vocab_size,
        "target_vocab_size": target_tokenizer.vocab_size,
        "max_source_length": MAX_SOURCE_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "start_token": START_TOKEN,
        "end_token": END_TOKEN,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }
    (output_dir / "retraining_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="models/retrained")
    parser.add_argument("--samples", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_and_save(
        args.output_dir,
        sample_count=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
