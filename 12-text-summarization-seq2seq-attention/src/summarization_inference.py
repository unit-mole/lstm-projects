"""Reusable greedy and beam-search inference for the saved Seq2Seq models."""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import keras
import numpy as np

from .attention_layer import decoder_step_with_attention
from .config import (
    DECODER_MODEL_PATH,
    ENCODER_MODEL_PATH,
    END_TOKEN,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    MIN_INPUT_WORDS,
    MODEL_METADATA_PATH,
    SOURCE_TOKENIZER_PATH,
    START_TOKEN,
    TARGET_TOKENIZER_PATH,
)
from .sequence_generation import pad_sequences_post
from .text_preprocessing import validate_input_text
from .tokenizer_utils import PortableTokenizer


@dataclass
class SummarizationResult:
    """Structured output returned by the summarization pipeline."""

    input_text: str
    cleaned_text: str
    summary: str
    decoding_method: str
    input_word_count: int
    summary_word_count: int
    compression_ratio: float
    oov_ratio: float
    truncated: bool
    warning: str = ""
    attention_matrix: np.ndarray | None = None
    source_tokens: list[str] | None = None
    generated_tokens: list[str] | None = None

    def as_record(self) -> dict[str, object]:
        return {
            "input_text": self.input_text,
            "generated_summary": self.summary,
            "decoding_method": self.decoding_method,
            "input_word_count": self.input_word_count,
            "summary_word_count": self.summary_word_count,
            "compression_ratio": self.compression_ratio,
            "oov_ratio": self.oov_ratio,
            "truncated": self.truncated,
            "warning": self.warning,
        }


class Summarizer:
    """Load the supplied encoder/decoder artifacts and generate summaries."""

    def __init__(
        self,
        encoder_path: str | Path = ENCODER_MODEL_PATH,
        decoder_path: str | Path = DECODER_MODEL_PATH,
        source_tokenizer_path: str | Path = SOURCE_TOKENIZER_PATH,
        target_tokenizer_path: str | Path = TARGET_TOKENIZER_PATH,
        metadata_path: str | Path = MODEL_METADATA_PATH,
    ) -> None:
        self.encoder_model = keras.models.load_model(encoder_path, compile=False)
        self.decoder_model = keras.models.load_model(decoder_path, compile=False)
        self.source_tokenizer = PortableTokenizer.load(source_tokenizer_path)
        self.target_tokenizer = PortableTokenizer.load(target_tokenizer_path)
        self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

        sequence = self.metadata["sequence"]
        architecture = self.metadata["architecture"]
        self.max_source_length = int(sequence["max_source_length"])
        self.max_target_length = int(sequence["max_target_length"])
        self.latent_dimension = int(architecture["latent_dimension"])
        self.start_id = self.target_tokenizer.word_index[START_TOKEN]
        self.end_id = self.target_tokenizer.word_index[END_TOKEN]

    def _prepare_encoder_input(self, text: str) -> tuple[str, np.ndarray, float, bool, list[str]]:
        validation = validate_input_text(text, min_words=MIN_INPUT_WORDS)
        if not validation.is_valid:
            raise ValueError(validation.message)
        source_tokens = validation.cleaned_text.split()
        sequence = self.source_tokenizer.text_to_sequence(validation.cleaned_text)
        truncated = len(sequence) > self.max_source_length
        padded = pad_sequences_post([sequence], self.max_source_length)
        oov_ratio = self.source_tokenizer.oov_ratio(validation.cleaned_text)
        return validation.cleaned_text, padded, oov_ratio, truncated, source_tokens[: self.max_source_length]

    def _greedy_decode(
        self,
        encoder_input: np.ndarray,
        capture_attention: bool,
    ) -> tuple[list[int], np.ndarray | None, str]:
        encoder_outputs, state_h, state_c = self.encoder_model.predict(
            encoder_input, verbose=0
        )
        current_token = np.array([[self.start_id]], dtype="int32")
        generated_ids: list[int] = []
        attention_rows: list[np.ndarray] = []
        attention_warning = ""
        attention_enabled = capture_attention

        for _ in range(self.max_target_length - 1):
            scores = None
            if attention_enabled:
                try:
                    probabilities, state_h, state_c, scores = decoder_step_with_attention(
                        self.decoder_model,
                        current_token,
                        encoder_outputs,
                        state_h,
                        state_c,
                    )
                except Exception:
                    # Summary generation must remain available even if an optional
                    # attention visualization cannot be produced on a host backend.
                    attention_enabled = False
                    attention_rows.clear()
                    attention_warning = (
                        "The summary was generated successfully, but attention "
                        "visualization is unavailable in this runtime."
                    )
                    probabilities, state_h, state_c = self.decoder_model.predict(
                        [current_token, encoder_outputs, state_h, state_c], verbose=0
                    )
            else:
                probabilities, state_h, state_c = self.decoder_model.predict(
                    [current_token, encoder_outputs, state_h, state_c], verbose=0
                )

            sampled_id = int(np.argmax(np.asarray(probabilities)[0, -1, :]))
            if sampled_id in (0, self.end_id):
                break
            generated_ids.append(sampled_id)
            if scores is not None:
                attention_rows.append(
                    np.asarray(scores[0, 0, :], dtype=float)
                )
            current_token = np.array([[sampled_id]], dtype="int32")

        matrix = np.asarray(attention_rows) if attention_rows else None
        return generated_ids, matrix, attention_warning

    def _beam_search_decode(
        self,
        encoder_input: np.ndarray,
        beam_width: int,
        length_penalty: float = 0.7,
    ) -> list[int]:
        encoder_outputs, initial_h, initial_c = self.encoder_model.predict(
            encoder_input, verbose=0
        )
        # score, token_ids, next_token, state_h, state_c, finished
        beams: list[tuple[float, list[int], int, np.ndarray, np.ndarray, bool]] = [
            (0.0, [], self.start_id, initial_h, initial_c, False)
        ]

        def normalized_score(item):
            score, tokens, *_ = item
            length = max(len(tokens), 1)
            return score / (length**length_penalty)

        for _ in range(self.max_target_length - 1):
            candidates = []
            for score, tokens, next_token, state_h, state_c, finished in beams:
                if finished:
                    candidates.append((score, tokens, next_token, state_h, state_c, True))
                    continue
                token_array = np.array([[next_token]], dtype="int32")
                probabilities, new_h, new_c = self.decoder_model.predict(
                    [token_array, encoder_outputs, state_h, state_c], verbose=0
                )
                distribution = np.asarray(probabilities[0, -1, :], dtype=float)
                top_ids = np.argpartition(distribution, -beam_width)[-beam_width:]
                top_ids = top_ids[np.argsort(distribution[top_ids])[::-1]]
                for token_id in top_ids:
                    token_id = int(token_id)
                    probability = max(float(distribution[token_id]), 1e-12)
                    is_finished = token_id in (0, self.end_id)
                    new_tokens = list(tokens)
                    if not is_finished:
                        new_tokens.append(token_id)
                    candidates.append(
                        (
                            score + float(np.log(probability)),
                            new_tokens,
                            token_id,
                            new_h,
                            new_c,
                            is_finished,
                        )
                    )
            beams = sorted(candidates, key=normalized_score, reverse=True)[:beam_width]
            if all(item[-1] for item in beams):
                break
        return max(beams, key=normalized_score)[1]

    def summarize(
        self,
        text: str,
        decoding_method: str = "greedy",
        beam_width: int = 3,
        include_attention: bool = True,
    ) -> SummarizationResult:
        """Generate one summary and return diagnostics for the app."""

        cleaned, encoder_input, oov_ratio, truncated, source_tokens = self._prepare_encoder_input(text)
        method = decoding_method.lower().strip()
        attention_matrix: np.ndarray | None = None
        attention_warning = ""
        if method == "greedy":
            generated_ids, attention_matrix, attention_warning = self._greedy_decode(
                encoder_input, capture_attention=include_attention
            )
        elif method in {"beam", "beam search", "beam_search"}:
            generated_ids = self._beam_search_decode(
                encoder_input, beam_width=max(2, min(int(beam_width), 5))
            )
            method = "beam search"
        else:
            raise ValueError("decoding_method must be 'greedy' or 'beam search'.")

        excluded = {START_TOKEN, END_TOKEN, self.target_tokenizer.oov_token}
        summary = self.target_tokenizer.tokens_to_text(generated_ids, excluded)
        if not summary:
            summary = (
                "The model could not generate a reliable summary for this input. "
                "Try an in-domain sample or provide more context."
            )

        input_words = len(cleaned.split())
        summary_words = len(summary.split())
        compression = summary_words / input_words if input_words else 0.0
        warning_parts = []
        if attention_warning:
            warning_parts.append(attention_warning)
        if truncated:
            warning_parts.append(
                f"Only the first {self.max_source_length} cleaned tokens were used."
            )
        if oov_ratio >= 0.25:
            warning_parts.append(
                "A large share of the input is outside the small training vocabulary; review the output carefully."
            )

        generated_tokens = [
            self.target_tokenizer.index_word.get(token_id, "")
            for token_id in generated_ids
            if self.target_tokenizer.index_word.get(token_id, "") not in excluded
        ]
        if attention_matrix is not None:
            attention_matrix = attention_matrix[:, : len(source_tokens)]

        return SummarizationResult(
            input_text=text,
            cleaned_text=cleaned,
            summary=summary,
            decoding_method=method,
            input_word_count=input_words,
            summary_word_count=summary_words,
            compression_ratio=compression,
            oov_ratio=oov_ratio,
            truncated=truncated,
            warning=" ".join(warning_parts),
            attention_matrix=attention_matrix,
            source_tokens=source_tokens,
            generated_tokens=generated_tokens,
        )

    def summarize_batch(
        self,
        texts: Iterable[str],
        decoding_method: str = "greedy",
        beam_width: int = 3,
    ) -> list[SummarizationResult]:
        """Summarize multiple rows while retaining row-level validation messages."""

        results: list[SummarizationResult] = []
        for text in texts:
            try:
                result = self.summarize(
                    str(text),
                    decoding_method=decoding_method,
                    beam_width=beam_width,
                    include_attention=False,
                )
            except ValueError as exc:
                cleaned = str(text or "")
                result = SummarizationResult(
                    input_text=str(text or ""),
                    cleaned_text=cleaned,
                    summary="",
                    decoding_method=decoding_method,
                    input_word_count=len(cleaned.split()),
                    summary_word_count=0,
                    compression_ratio=0.0,
                    oov_ratio=0.0,
                    truncated=False,
                    warning=str(exc),
                )
            results.append(result)
        return results
