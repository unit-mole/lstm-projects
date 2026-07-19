from __future__ import annotations
import re

def clean_text(text: object) -> str:
    """Apply the exact normalization used by the supplied model."""
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def validate_user_text(text: object, max_characters: int = 500) -> str:
    value = str(text or "").strip()
    if not value:
        raise ValueError("Enter a message before generating a response.")
    if len(value) > max_characters:
        raise ValueError(f"Keep the message below {max_characters} characters.")
    cleaned = clean_text(value)
    if not cleaned:
        raise ValueError("The message did not contain usable letters or numbers.")
    return cleaned
