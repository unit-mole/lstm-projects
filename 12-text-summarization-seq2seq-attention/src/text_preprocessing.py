"""Text cleaning and input-quality checks used by training and inference."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextValidation:
    """Validation result for a candidate summarization input."""

    is_valid: bool
    cleaned_text: str
    word_count: int
    message: str = ""


def clean_text(text: object) -> str:
    """Match the uploaded notebook's conservative training-time cleaning.

    The supplied model was trained on lowercase ASCII letters, digits, and
    whitespace. The same transformation is therefore required at inference.
    """

    value = str(text or "").lower()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def count_words(text: object) -> int:
    """Return a whitespace-token word count after cleaning."""

    cleaned = clean_text(text)
    return len(cleaned.split()) if cleaned else 0


def validate_input_text(text: object, min_words: int = 8) -> TextValidation:
    """Validate empty and extremely short inputs without crashing the app."""

    cleaned = clean_text(text)
    words = len(cleaned.split()) if cleaned else 0
    if not cleaned:
        return TextValidation(False, cleaned, words, "Enter text before generating a summary.")
    if words < min_words:
        return TextValidation(
            False,
            cleaned,
            words,
            "The input is too short for a reliable summary. Use at least "
            f"{min_words} words for this portfolio model.",
        )
    return TextValidation(True, cleaned, words)
