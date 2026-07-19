from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .text_preprocessing import clean_text

INPUT_ALIASES = ("input_text", "input", "question", "prompt", "message", "user_message")
TARGET_ALIASES = ("target_text", "response", "answer", "reply", "target", "bot_response")

def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None

def prepare_conversation_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("The conversation dataset is empty.")
    input_column = _find_column(list(frame.columns), INPUT_ALIASES)
    target_column = _find_column(list(frame.columns), TARGET_ALIASES)
    if input_column is None or target_column is None:
        raise ValueError(
            "Could not identify input and response columns. "
            f"Supported input aliases: {INPUT_ALIASES}; response aliases: {TARGET_ALIASES}."
        )
    prepared = frame[[input_column, target_column]].rename(
        columns={input_column: "input_text", target_column: "target_text"}
    )
    prepared = prepared.dropna().copy()
    prepared["input_text"] = prepared["input_text"].astype(str).str.strip()
    prepared["target_text"] = prepared["target_text"].astype(str).str.strip()
    prepared = prepared[prepared["input_text"].ne("") & prepared["target_text"].ne("")].copy()
    prepared["input_clean"] = prepared["input_text"].map(clean_text)
    prepared["target_clean"] = prepared["target_text"].map(clean_text)
    prepared = prepared[prepared["input_clean"].ne("") & prepared["target_clean"].ne("")]
    prepared = prepared.drop_duplicates(subset=["input_clean", "target_clean"]).reset_index(drop=True)
    if prepared.empty:
        raise ValueError("No usable input-response pairs remained after cleaning.")
    return prepared

def load_conversation_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        raw = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    else:
        raise ValueError("Only CSV and JSON conversation files are supported.")
    return prepare_conversation_pairs(raw)
