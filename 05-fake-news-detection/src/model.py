from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int
    embedding_dim: int = 64
    hidden_dim: int = 40
    dense_dim: int = 64
    dropout: float = 0.35
    padding_index: int = 0
    bidirectional: bool = True
    pooling: str = "masked_mean_max"

    def to_dict(self) -> dict:
        return asdict(self)


class FakeNewsLSTM(nn.Module):
    """Bidirectional LSTM with padding-aware mean and max pooling."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        directions = 2 if config.bidirectional else 1
        recurrent_size = config.hidden_dim * directions

        self.embedding = nn.Embedding(
            num_embeddings=config.vocabulary_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_index,
        )
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(recurrent_size * 2, config.dense_dim),
            nn.ReLU(),
            nn.Dropout(max(config.dropout - 0.10, 0.0)),
            nn.Linear(config.dense_dim, 1),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        sequence_output, _ = self.lstm(embedded)

        valid_mask = token_ids.ne(self.config.padding_index)
        expanded_mask = valid_mask.unsqueeze(-1)

        valid_counts = expanded_mask.sum(dim=1).clamp_min(1)
        mean_pool = (sequence_output * expanded_mask).sum(dim=1) / valid_counts

        negative_large = torch.finfo(sequence_output.dtype).min
        max_pool = sequence_output.masked_fill(~expanded_mask, negative_large).max(dim=1).values
        all_padding = ~valid_mask.any(dim=1)
        if all_padding.any():
            max_pool = max_pool.masked_fill(all_padding.unsqueeze(-1), 0.0)

        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(pooled).squeeze(-1)
