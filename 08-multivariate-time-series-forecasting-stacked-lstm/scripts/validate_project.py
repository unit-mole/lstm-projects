from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "app/streamlit_app.py",
    "data/hourly_energy.csv",
    "models/stacked_lstm_energy.keras",
    "models/scalers.json",
    "models/model_metadata.json",
    "outputs/model_metrics.json",
    "README.md",
    "requirements.txt",
]

missing = [path for path in REQUIRED if not (ROOT / path).exists()]
if missing:
    raise SystemExit(f"Missing required project files: {missing}")

metadata = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
data = pd.read_csv(ROOT / "data/hourly_energy.csv")
if len(data) != metadata["record_count"]:
    raise SystemExit("Data record count does not match metadata.")
if metadata["input_shape"] != [24, 8]:
    raise SystemExit("Unexpected model input shape.")
print("Project validation passed.")
