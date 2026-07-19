"""High-level convenience functions for application and API-style usage."""

from __future__ import annotations

from functools import lru_cache

from .summarization_inference import SummarizationResult, Summarizer


@lru_cache(maxsize=1)
def get_summarizer() -> Summarizer:
    """Load model artifacts once per Python process."""

    return Summarizer()


def generate_summary(
    input_text: str,
    decoding_method: str = "greedy",
    beam_width: int = 3,
) -> SummarizationResult:
    """Generate a summary with the default saved artifacts."""

    return get_summarizer().summarize(
        input_text,
        decoding_method=decoding_method,
        beam_width=beam_width,
    )
