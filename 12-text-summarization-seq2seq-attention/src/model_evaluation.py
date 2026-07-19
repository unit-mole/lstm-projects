"""ROUGE, BLEU-like, and baseline evaluation helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from rouge_score import rouge_scorer


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def simple_bleu(reference: str, candidate: str, max_n: int = 2) -> float:
    """Reproduce the notebook's transparent BLEU-like overlap score."""

    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    if not candidate_tokens:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        reference_counts = Counter(_ngrams(reference_tokens, n))
        candidate_counts = Counter(_ngrams(candidate_tokens, n))
        overlap = sum(
            min(count, reference_counts[ngram])
            for ngram, count in candidate_counts.items()
        )
        precisions.append(overlap / max(sum(candidate_counts.values()), 1))
    brevity_penalty = min(
        1.0,
        np.exp(1 - len(reference_tokens) / max(len(candidate_tokens), 1)),
    )
    return float(
        brevity_penalty
        * np.prod([max(value, 1e-8) for value in precisions]) ** (1 / max_n)
    )


def compute_rouge(reference: str, generated: str) -> dict[str, float]:
    """Return ROUGE-1, ROUGE-2, and ROUGE-L F1 values."""

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, generated)
    return {
        "rouge_1_f1": float(scores["rouge1"].fmeasure),
        "rouge_2_f1": float(scores["rouge2"].fmeasure),
        "rouge_l_f1": float(scores["rougeL"].fmeasure),
    }


def evaluate_summaries(
    references: Iterable[str],
    predictions: Iterable[str],
) -> dict[str, float]:
    references = list(references)
    predictions = list(predictions)
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have the same length.")
    if not references:
        raise ValueError("At least one reference/prediction pair is required.")
    rouge = [compute_rouge(ref, pred) for ref, pred in zip(references, predictions)]
    return {
        "rouge_1_f1": float(np.mean([row["rouge_1_f1"] for row in rouge])),
        "rouge_2_f1": float(np.mean([row["rouge_2_f1"] for row in rouge])),
        "rouge_l_f1": float(np.mean([row["rouge_l_f1"] for row in rouge])),
        "bleu_like": float(np.mean([simple_bleu(ref, pred) for ref, pred in zip(references, predictions)])),
        "exact_match_ratio": float(np.mean([ref.strip() == pred.strip() for ref, pred in zip(references, predictions)])),
    }
