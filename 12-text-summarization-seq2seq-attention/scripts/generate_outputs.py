"""Regenerate evaluation tables and a sample attention visualization."""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, SAMPLE_ARTICLES_PATH
from src.model_evaluation import compute_rouge
from src.summarization_inference import Summarizer
from src.visualization import create_attention_heatmap


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summarizer = Summarizer()
    samples = pd.read_csv(SAMPLE_ARTICLES_PATH)
    records = []
    first_result = None
    for _, row in samples.iterrows():
        result = summarizer.summarize(
            row["input_text"], decoding_method="greedy", include_attention=True
        )
        if first_result is None:
            first_result = result
        record = result.as_record()
        record["sample_id"] = row["sample_id"]
        record["reference_summary"] = row["target_summary"]
        record.update(compute_rouge(row["target_summary"].lower().strip(" ."), result.summary))
        records.append(record)

    output = pd.DataFrame(records)
    output.to_csv(OUTPUT_DIR / "sample_app_predictions.csv", index=False)
    summary = {
        "rows": int(len(output)),
        "mean_rouge_1_f1": float(output["rouge_1_f1"].mean()),
        "mean_rouge_2_f1": float(output["rouge_2_f1"].mean()),
        "mean_rouge_l_f1": float(output["rouge_l_f1"].mean()),
    }
    (OUTPUT_DIR / "sample_app_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if (
        first_result is not None
        and first_result.attention_matrix is not None
        and first_result.source_tokens
        and first_result.generated_tokens
    ):
        figure = create_attention_heatmap(
            first_result.attention_matrix,
            first_result.source_tokens,
            first_result.generated_tokens,
        )
        figure.savefig(
            OUTPUT_DIR / "attention_visualization_regenerated.png",
            dpi=180,
            bbox_inches="tight",
        )
    print(f"Generated outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
