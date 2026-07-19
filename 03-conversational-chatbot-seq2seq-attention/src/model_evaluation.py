from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd

def _ngrams(tokens: list[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def simple_bleu(reference: str, candidate: str, max_n: int = 2) -> float:
    ref_tokens, cand_tokens = reference.split(), candidate.split()
    if not cand_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = Counter(_ngrams(ref_tokens, n))
        cand_counts = Counter(_ngrams(cand_tokens, n))
        overlap = sum(min(count, ref_counts[ngram]) for ngram, count in cand_counts.items())
        precisions.append(overlap / max(sum(cand_counts.values()), 1))
    bp = min(1.0, np.exp(1.0 - len(ref_tokens) / max(len(cand_tokens), 1)))
    return float(bp * np.prod([max(p, 1e-8) for p in precisions]) ** (1.0 / max_n))

def evaluate_responses(frame: pd.DataFrame):
    required = {"reference_response", "predicted_response"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Evaluation frame must contain {required}.")
    bleu = [simple_bleu(r, p) for r, p in zip(frame.reference_response, frame.predicted_response)]
    exact = frame.reference_response.str.strip() == frame.predicted_response.str.strip()
    return {
        "bleu_like_mean": float(np.mean(bleu)),
        "exact_match_ratio": float(exact.mean()),
        "average_reference_length": float(frame.reference_response.str.split().str.len().mean()),
        "average_predicted_length": float(frame.predicted_response.str.split().str.len().mean()),
    }
