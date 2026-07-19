from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from .text_preprocessing import tokenize_text


def _save(path: str | Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_distribution(frame: pd.DataFrame, path: str | Path) -> None:
    counts = frame["display_label"].value_counts().reindex(["Real", "Fake"])
    counts.plot(kind="bar", title="LIAR Binary Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Statements")
    _save(path)


def plot_length_distribution(frame: pd.DataFrame, path: str | Path) -> None:
    lengths = frame["statement"].map(lambda value: len(tokenize_text(value)))
    lengths.plot(kind="hist", bins=30, title="Statement Length Distribution")
    plt.xlabel("Tokens")
    _save(path)


def plot_training_history(history: list[dict], path: str | Path) -> None:
    data = pd.DataFrame(history).copy()
    if "training_loss" not in data.columns and "loss" in data.columns:
        data = data.rename(columns={"loss": "training_loss"})
    available = [column for column in ["training_loss", "macro_f1", "roc_auc"] if column in data.columns]
    if not available:
        raise ValueError("Training history does not contain plottable metrics.")
    data.plot(x="epoch", y=available, marker="o")
    plt.title("Training History")
    plt.xlabel("Epoch")
    _save(path)


def plot_confusion_matrix(matrix: list[list[int]], path: str | Path) -> None:
    values = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(values)
    axis.set_xticks([0, 1], labels=["Predicted Real", "Predicted Fake"])
    axis.set_yticks([0, 1], labels=["Actual Real", "Actual Fake"])
    axis.set_title("LSTM Confusion Matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    _save(path)


def plot_roc_and_pr(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    roc_path: str | Path,
    pr_path: str | Path,
) -> None:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    RocCurveDisplay.from_predictions(y, p)
    plt.title("LSTM ROC Curve")
    _save(roc_path)
    PrecisionRecallDisplay.from_predictions(y, p)
    plt.title("LSTM Precision-Recall Curve")
    _save(pr_path)


def plot_model_comparison(metrics: dict[str, dict], path: str | Path) -> None:
    table = pd.DataFrame(metrics).T[["accuracy", "f1", "macro_f1", "roc_auc", "pr_auc"]]
    table.plot(kind="bar", title="Baseline vs LSTM")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    _save(path)


def plot_frequent_words(frame: pd.DataFrame, path: str | Path, top_n: int = 15) -> None:
    records = {}
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "is", "are",
        "was", "were", "that", "this", "with", "has", "have", "had", "says", "said",
    }
    for label in ["Real", "Fake"]:
        counts: Counter[str] = Counter()
        for text in frame.loc[frame["display_label"] == label, "statement"]:
            counts.update(token for token in tokenize_text(text) if token not in stop and token.isalpha())
        records[label] = dict(counts.most_common(top_n))

    combined = pd.DataFrame(records).fillna(0)
    combined.plot(kind="bar", figsize=(11, 5), title="Frequent Words by Class")
    plt.ylabel("Frequency")
    _save(path)
