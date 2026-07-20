import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import pandas as pd

from src.config import SAMPLE_ARTICLES_PATH
from src.summarization_inference import Summarizer


def test_saved_model_generates_nonempty_summary():
    sample = pd.read_csv(SAMPLE_ARTICLES_PATH).iloc[0]
    summarizer = Summarizer()
    result = summarizer.summarize(
        sample["input_text"], decoding_method="greedy", include_attention=True
    )
    assert result.summary
    assert result.summary_word_count > 0
    assert result.attention_matrix is not None
    assert result.attention_matrix.shape[0] == len(result.generated_tokens)
    assert result.attention_matrix.shape[1] == len(result.source_tokens)


def test_batch_validation_does_not_crash_on_empty_text():
    summarizer = Summarizer()
    results = summarizer.summarize_batch([""])
    assert results[0].summary == ""
    assert results[0].warning


def test_attention_runtime_failure_falls_back_to_summary(monkeypatch):
    """Optional attention rendering must never block summary generation."""

    import src.summarization_inference as inference_module

    sample = pd.read_csv(SAMPLE_ARTICLES_PATH).iloc[0]
    summarizer = Summarizer()

    def fail_attention_step(*args, **kwargs):
        raise ValueError("simulated hosted-backend graph failure")

    monkeypatch.setattr(
        inference_module,
        "decoder_step_with_attention",
        fail_attention_step,
    )
    result = summarizer.summarize(
        sample["input_text"], decoding_method="greedy", include_attention=True
    )

    assert result.summary
    assert result.attention_matrix is None
    assert "attention visualization is unavailable" in result.warning.lower()
