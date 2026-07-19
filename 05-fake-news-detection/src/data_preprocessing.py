from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "job_title",
    "state_info",
    "party_affiliation",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]

BINARY_LABEL_MAP = {
    "pants-fire": 1,
    "false": 1,
    "barely-true": 1,
    "half-true": 0,
    "mostly-true": 0,
    "true": 0,
}

DISPLAY_LABELS = {0: "Real", 1: "Fake"}


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    integrity_report: dict


def _normalized_key(text: object) -> str:
    normalized = " ".join(str(text).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _prepare_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "statement" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"{split_name} split must contain 'statement' and 'label' columns.")

    prepared = frame.copy()
    prepared["statement"] = prepared["statement"].fillna("").astype(str).str.strip()
    prepared["label_name"] = prepared["label"].astype(str).str.lower().str.strip()
    prepared["target"] = prepared["label_name"].map(BINARY_LABEL_MAP)
    prepared = prepared.loc[(prepared["statement"] != "") & prepared["target"].notna()].copy()
    prepared["target"] = prepared["target"].astype(int)
    prepared["display_label"] = prepared["target"].map(DISPLAY_LABELS)
    prepared["text_key"] = prepared["statement"].map(_normalized_key)
    prepared = prepared.drop_duplicates(subset="text_key", keep="first").reset_index(drop=True)
    prepared["split"] = split_name
    return prepared


def _remove_cross_split_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> DatasetSplits:
    train_keys = set(train["text_key"])
    validation_before = len(validation)
    validation = validation.loc[~validation["text_key"].isin(train_keys)].copy()

    protected_keys = train_keys | set(validation["text_key"])
    test_before = len(test)
    test = test.loc[~test["text_key"].isin(protected_keys)].copy()

    report = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_overlap_removed": validation_before - len(validation),
        "test_overlap_removed": test_before - len(test),
        "train_class_counts": train["display_label"].value_counts().to_dict(),
        "validation_class_counts": validation["display_label"].value_counts().to_dict(),
        "test_class_counts": test["display_label"].value_counts().to_dict(),
    }
    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        integrity_report=report,
    )


def load_local_liar(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
) -> DatasetSplits:
    def read_tsv(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=LIAR_COLUMNS,
            quoting=csv.QUOTE_NONE,
            engine="python",
            on_bad_lines="skip",
        )

    train = _prepare_frame(read_tsv(train_path), "train")
    validation = _prepare_frame(read_tsv(validation_path), "validation")
    test = _prepare_frame(read_tsv(test_path), "test")
    return _remove_cross_split_overlap(train, validation, test)


def load_huggingface_liar() -> DatasetSplits:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for Hugging Face loading. "
            "Install requirements-train.txt."
        ) from exc

    dataset = load_dataset("ucsbnlp/liar")

    def convert(split_name: str, target_name: str) -> pd.DataFrame:
        split = dataset[split_name]
        frame = split.to_pandas()
        feature = split.features.get("label")
        if feature is not None and hasattr(feature, "int2str"):
            frame["label"] = frame["label"].map(lambda value: feature.int2str(int(value)))
        return _prepare_frame(frame, target_name)

    train = convert("train", "train")
    validation = convert("validation", "validation")
    test = convert("test", "test")
    return _remove_cross_split_overlap(train, validation, test)
