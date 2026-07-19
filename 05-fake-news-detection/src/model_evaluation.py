from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
) -> dict[str, float | list[list[int]]]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    predictions = (p >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predictions, average="binary", zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y, predictions, average="macro", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }


def select_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    minimum: float = 0.30,
    maximum: float = 0.70,
    step: float = 0.005,
) -> tuple[float, dict]:
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(probabilities), dtype=float)
    candidates = np.arange(minimum, maximum + step / 2, step)
    scored = [(float(t), classification_metrics(y, p, float(t))) for t in candidates]
    threshold, metrics = max(
        scored,
        key=lambda item: (
            item[1]["macro_f1"],
            item[1]["balanced_accuracy"],
            -abs(item[0] - 0.5),
        ),
    )
    return threshold, metrics


def train_tfidf_baseline(
    train_texts: Iterable[str],
    train_labels: Iterable[int],
    evaluation_texts: Iterable[str],
    random_state: int = 42,
):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=30_000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(list(train_texts))
    x_eval = vectorizer.transform(list(evaluation_texts))
    model = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        C=1.5,
        solver="liblinear",
        random_state=random_state,
    )
    model.fit(x_train, np.asarray(list(train_labels), dtype=int))
    probabilities = model.predict_proba(x_eval)[:, 1]
    return vectorizer, model, probabilities


def make_prediction_frame(
    source: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    output = source[["statement", "label_name", "display_label", "target"]].copy()
    output["fake_probability"] = probabilities
    output["predicted_target"] = (probabilities >= threshold).astype(int)
    output["predicted_label"] = output["predicted_target"].map({0: "Real", 1: "Fake"})
    output["confidence"] = np.maximum(probabilities, 1 - probabilities)
    output["correct"] = output["predicted_target"].eq(output["target"])
    output["error_type"] = "correct"
    output.loc[(output["target"] == 0) & (output["predicted_target"] == 1), "error_type"] = (
        "false_positive_real_as_fake"
    )
    output.loc[(output["target"] == 1) & (output["predicted_target"] == 0), "error_type"] = (
        "false_negative_fake_as_real"
    )
    return output


def save_json(payload: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
