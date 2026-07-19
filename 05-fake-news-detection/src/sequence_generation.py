from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .text_preprocessing import TokenizerConfig, Vocabulary, encode_texts


@dataclass
class SequenceBundle:
    tokens: torch.Tensor
    labels: torch.Tensor


def create_sequence_bundle(
    texts: Sequence[object],
    labels: Sequence[int] | np.ndarray,
    vocabulary: Vocabulary,
    tokenizer_config: TokenizerConfig,
) -> SequenceBundle:
    token_array = encode_texts(texts, vocabulary, tokenizer_config)
    label_array = np.asarray(labels, dtype=np.float32)
    return SequenceBundle(
        tokens=torch.as_tensor(token_array, dtype=torch.long),
        labels=torch.as_tensor(label_array, dtype=torch.float32),
    )


def create_data_loader(
    bundle: SequenceBundle,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(bundle.tokens, bundle.labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )
