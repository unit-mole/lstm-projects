"""Sequence padding and teacher-forcing preparation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def pad_sequences_post(
    sequences: Iterable[Iterable[int]],
    max_length: int,
    dtype: str = "int32",
) -> np.ndarray:
    """Post-pad and post-truncate integer sequences."""

    values = [list(sequence) for sequence in sequences]
    padded = np.zeros((len(values), max_length), dtype=dtype)
    for row_index, sequence in enumerate(values):
        truncated = sequence[:max_length]
        padded[row_index, : len(truncated)] = truncated
    return padded


def prepare_teacher_forcing_targets(target_sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift targets into decoder inputs and one-step-ahead decoder labels."""

    if target_sequences.ndim != 2 or target_sequences.shape[1] < 2:
        raise ValueError("Target sequences must have shape [samples, time] with time >= 2.")
    return target_sequences[:, :-1], target_sequences[:, 1:]
