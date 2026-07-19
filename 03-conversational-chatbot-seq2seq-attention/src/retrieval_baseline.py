from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from .text_preprocessing import clean_text

@dataclass
class RetrievalResult:
    matched_input: str
    response: str
    similarity: float

def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(clean_text(left).split())
    right_tokens = set(clean_text(right).split())
    union = left_tokens | right_tokens
    return 0.0 if not union else len(left_tokens & right_tokens) / len(union)

def retrieve_response(text: str, conversations: pd.DataFrame) -> RetrievalResult:
    if conversations.empty:
        raise ValueError("The retrieval corpus is empty.")
    scores = conversations["input_text"].map(lambda candidate: _token_jaccard(text, candidate))
    best_index = int(scores.idxmax())
    row = conversations.loc[best_index]
    return RetrievalResult(str(row["input_text"]), str(row["target_text"]), float(scores.loc[best_index]))
