"""Validate required project artifacts without loading the ML backend."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "app/streamlit_app.py",
    "app/requirements.txt",
    "data/sample_articles.csv",
    "data/sample_batch.csv",
    "models/seq2seq_summarization.keras",
    "models/encoder_summarization.keras",
    "models/decoder_summarization.keras",
    "models/source_tokenizer.json",
    "models/target_tokenizer.json",
    "models/model_metadata.json",
    "models/model_metrics.json",
    "src/summarization_inference.py",
    "README.md",
    "README_HOSTING.md",
]


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (PROJECT_ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    metadata = json.loads(
        (PROJECT_ROOT / "models/model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["sequence"]["source_vocab_size"] == 88
    assert metadata["sequence"]["target_vocab_size"] == 57
    assert metadata["sequence"]["max_source_length"] == 49
    assert metadata["sequence"]["max_target_length"] == 12

    source = json.loads(
        (PROJECT_ROOT / "models/source_tokenizer.json").read_text(encoding="utf-8")
    )
    target = json.loads(
        (PROJECT_ROOT / "models/target_tokenizer.json").read_text(encoding="utf-8")
    )
    assert source["vocab_size"] == 88
    assert target["vocab_size"] == 57
    assert target["word_index"]["sostok"] == 2
    assert target["word_index"]["eostok"] == 6

    samples = pd.read_csv(PROJECT_ROOT / "data/sample_articles.csv")
    required_columns = {"sample_id", "title", "input_text", "target_summary"}
    assert required_columns.issubset(samples.columns)
    assert len(samples) >= 5

    for model_name in [
        "seq2seq_summarization.keras",
        "encoder_summarization.keras",
        "decoder_summarization.keras",
    ]:
        model_path = PROJECT_ROOT / "models" / model_name
        with zipfile.ZipFile(model_path) as archive:
            names = set(archive.namelist())
            assert {"config.json", "metadata.json", "model.weights.h5"}.issubset(names)

    print("Project validation passed.")


if __name__ == "__main__":
    main()
