from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "app/streamlit_app.py", "app/requirements.txt",
    "data/sample_weather_sequences.npz", "models/convlstm_weather_forecast.keras",
    "models/model_metadata.json", "notebooks/weather_forecasting_convlstm.ipynb",
    "requirements.txt", "train_model.py",
]

missing = [path for path in REQUIRED if not (ROOT / path).exists()]
if missing:
    raise SystemExit(f"Missing required files: {missing}")
metadata = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
assert metadata["input_shape"] == [6, 24, 24, 1]
with zipfile.ZipFile(ROOT / "models/convlstm_weather_forecast.keras") as archive:
    names = set(archive.namelist())
    assert {"config.json", "model.weights.h5", "metadata.json"}.issubset(names)
print("Project validation passed")
