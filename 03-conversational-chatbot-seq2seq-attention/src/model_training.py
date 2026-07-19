from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .data_preprocessing import prepare_conversation_pairs

def _require_keras():
    os.environ.setdefault("KERAS_BACKEND", "jax")
    try:
        import keras
        from keras.src.legacy.preprocessing.text import Tokenizer
        from keras.src.utils.sequence_utils import pad_sequences
    except ImportError as exc:
        raise ImportError("Install requirements.txt before retraining the chatbot.") from exc
    return keras, Tokenizer, pad_sequences

def group_aware_split(frame: pd.DataFrame, seed: int = 42):
    # Split unique input-response pairs so exact pairs cannot cross splits.
    unique_pairs = frame.drop_duplicates(subset=["input_clean", "target_clean"]).reset_index(drop=True)
    if len(unique_pairs) < 10:
        raise ValueError("At least 10 unique dialogue pairs are recommended for retraining.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_pairs))
    train_end = max(1, int(len(order) * 0.70))
    validation_end = max(train_end + 1, int(len(order) * 0.85))
    unique_pairs["_pair_id"] = np.arange(len(unique_pairs))
    train = unique_pairs[unique_pairs["_pair_id"].isin(set(order[:train_end].tolist()))]
    validation = unique_pairs[unique_pairs["_pair_id"].isin(set(order[train_end:validation_end].tolist()))]
    test = unique_pairs[unique_pairs["_pair_id"].isin(set(order[validation_end:].tolist()))]
    return tuple(part.drop(columns="_pair_id").reset_index(drop=True) for part in (train, validation, test))

def build_model(source_vocab_size: int, target_vocab_size: int, max_source_length: int,
                max_target_length: int, embedding_dim: int = 128, latent_dim: int = 128):
    keras, _, _ = _require_keras()
    encoder_inputs = keras.Input(shape=(max_source_length,), name="encoder_inputs")
    encoder_embedding = keras.layers.Embedding(
        source_vocab_size, embedding_dim, mask_zero=True, name="encoder_embedding"
    )(encoder_inputs)
    encoder_lstm = keras.layers.LSTM(
        latent_dim, return_sequences=True, return_state=True, name="encoder_lstm"
    )
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    decoder_inputs = keras.Input(shape=(max_target_length - 1,), name="decoder_inputs")
    decoder_embedding = keras.layers.Embedding(
        target_vocab_size, embedding_dim, mask_zero=True, name="decoder_embedding"
    )(decoder_inputs)
    decoder_lstm = keras.layers.LSTM(
        latent_dim, return_sequences=True, return_state=True, name="decoder_lstm"
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
    attention_output = keras.layers.AdditiveAttention(name="attention_layer")(
        [decoder_outputs, encoder_outputs]
    )
    combined = keras.layers.Concatenate(axis=-1)([decoder_outputs, attention_output])
    predictions = keras.layers.TimeDistributed(
        keras.layers.Dense(target_vocab_size, activation="softmax"),
        name="time_distributed_output",
    )(combined)
    model = keras.Model([encoder_inputs, decoder_inputs], predictions, name="seq2seq_attention_chatbot")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_chatbot(frame: pd.DataFrame, output_dir: str | Path, epochs: int = 40,
                  batch_size: int = 32, seed: int = 42) -> dict[str, Any]:
    keras, Tokenizer, pad_sequences = _require_keras()
    output_dir = Path(output_dir)
    models_dir, outputs_dir = output_dir/"models", output_dir/"outputs"
    models_dir.mkdir(parents=True, exist_ok=True); outputs_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_conversation_pairs(frame)
    prepared["target_seq"] = "sostok " + prepared["target_clean"] + " eostok"
    train, validation, test = group_aware_split(prepared, seed=seed)
    source_tokenizer = Tokenizer(oov_token="<unk>"); source_tokenizer.fit_on_texts(train["input_clean"])
    target_tokenizer = Tokenizer(oov_token="<unk>"); target_tokenizer.fit_on_texts(train["target_seq"])
    max_source_length = int(max(train["input_clean"].str.split().str.len().max(), 1))
    max_target_length = int(max(train["target_seq"].str.split().str.len().max(), 3))
    source_vocab_size = len(source_tokenizer.word_index) + 1
    target_vocab_size = len(target_tokenizer.word_index) + 1
    def prepare_split(split):
        source = pad_sequences(source_tokenizer.texts_to_sequences(split["input_clean"]),
                               maxlen=max_source_length, padding="post")
        target = pad_sequences(target_tokenizer.texts_to_sequences(split["target_seq"]),
                               maxlen=max_target_length, padding="post")
        return source, target[:, :-1], target[:, 1:]
    train_x, train_dec, train_y = prepare_split(train)
    val_x, val_dec, val_y = prepare_split(validation)
    test_x, test_dec, test_y = prepare_split(test)
    model = build_model(source_vocab_size, target_vocab_size, max_source_length, max_target_length)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]
    history = model.fit(
        [train_x, train_dec], train_y[..., np.newaxis],
        validation_data=([val_x, val_dec], val_y[..., np.newaxis]),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2
    )
    test_loss, test_accuracy = model.evaluate(
        [test_x, test_dec], test_y[..., np.newaxis], verbose=0
    )
    model.save(models_dir/"seq2seq_chatbot_retrained.keras")
    pd.DataFrame(history.history).assign(
        epoch=lambda frame_: np.arange(1, len(frame_) + 1)
    ).to_csv(outputs_dir/"retraining_history.csv", index=False)
    summary = {
        "test_loss": float(test_loss), "test_token_accuracy": float(test_accuracy),
        "source_vocab_size": int(source_vocab_size), "target_vocab_size": int(target_vocab_size),
        "max_source_length": int(max_source_length), "max_target_length": int(max_target_length),
        "split_strategy": "unique_pair_group_split", "seed": int(seed),
    }
    (models_dir/"retraining_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
