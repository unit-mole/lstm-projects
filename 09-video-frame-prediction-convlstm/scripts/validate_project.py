"""Fast filesystem and artifact validation before pushing the project to GitHub."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "app/streamlit_app.py",
    "data/sample_sequences.npz",
    "models/convlstm_video_prediction.keras",
    "models/model_metadata.json",
    "notebooks/video_frame_prediction_convlstm.ipynb",
    "requirements.txt",
    "src/prediction_pipeline.py",
]


def main() -> None:
    missing = [item for item in REQUIRED if not (ROOT / item).exists()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))
    metadata = json.loads((ROOT / "models/model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["input_frames"] == 6
    assert metadata["height"] == 32 and metadata["width"] == 32
    print("Project validation passed.")


if __name__ == "__main__":
    main()
