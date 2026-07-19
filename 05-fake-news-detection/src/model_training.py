from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import FakeNewsLSTM, ModelConfig
from .model_evaluation import classification_metrics, select_threshold


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 0.0015
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    patience: int = 3
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def positive_class_weight(labels: Iterable[int]) -> float:
    y = np.asarray(list(labels), dtype=int)
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    return float(negatives / positives)


def predict_probabilities(
    model: FakeNewsLSTM,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for token_ids, target in loader:
            token_ids = token_ids.to(device)
            logits = model(token_ids)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(target.numpy().astype(int).tolist())
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def train_model(
    model: FakeNewsLSTM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    training_labels: Iterable[int],
    config: TrainingConfig,
    device: torch.device | None = None,
) -> tuple[FakeNewsLSTM, float, dict, list[dict]]:
    set_reproducible_seed(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    weight = torch.tensor([positive_class_weight(training_labels)], dtype=torch.float32, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    best_threshold = 0.5
    best_validation_metrics: dict = {}
    history: list[dict] = []
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for token_ids, target in train_loader:
            token_ids = token_ids.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
            loss = loss_function(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        y_validation, p_validation = predict_probabilities(model, validation_loader, device)
        threshold, metrics = select_threshold(y_validation, p_validation)
        epoch_record = {
            "epoch": epoch,
            "training_loss": float(np.mean(batch_losses)),
            **metrics,
        }
        history.append(epoch_record)

        score = float(metrics["macro_f1"] + metrics["roc_auc"])
        if score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_threshold = threshold
            best_validation_metrics = metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model = model.cpu()
    return model, best_threshold, best_validation_metrics, history


def save_checkpoint(
    model: FakeNewsLSTM,
    model_config: ModelConfig,
    threshold: float,
    path: str | Path,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config.to_dict(),
        "threshold": float(threshold),
        "format_version": 1,
    }
    torch.save(checkpoint, Path(path))
