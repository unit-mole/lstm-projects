from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_pipeline import PredictiveMaintenancePipeline
from src.model_evaluation import evaluate_labeled_scores
from src.synthetic_data import generate_predictive_maintenance_data

pipeline = PredictiveMaintenancePipeline.from_artifacts(PROJECT_ROOT / "models")
frame, _ = generate_predictive_maintenance_data()
test = frame[frame["unit_id"].between(103, 120)]
result = pipeline.score_dataframe(test)
metrics = evaluate_labeled_scores(
    result.predictions["true_label"].to_numpy(),
    result.predictions["reconstruction_error"].to_numpy(),
    float(pipeline.metadata["threshold"]),
)
output = PROJECT_ROOT / "outputs" / "artifact_recheck_metrics.json"
output.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print(f"Saved to {output}")
