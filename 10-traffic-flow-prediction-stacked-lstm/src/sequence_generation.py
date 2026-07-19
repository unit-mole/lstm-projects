"""Chronological supervised-sequence construction for Stacked LSTM models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the previous sequence_length rows to predict the next target."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same row count.")
    if len(features) <= sequence_length:
        raise ValueError(
            "The dataset must contain more rows than the sequence length."
        )

    sequences = []
    labels = []
    for index in range(sequence_length, len(features)):
        sequences.append(features[index - sequence_length : index])
        labels.append(targets[index])

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def sequence_target_timestamps(
    timestamps: pd.Series,
    sequence_length: int,
) -> pd.Series:
    """Return timestamps corresponding to sequence targets."""
    return timestamps.iloc[sequence_length:].reset_index(drop=True)
