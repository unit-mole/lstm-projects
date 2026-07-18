"""Validate that the project contains the files required for testing and deployment."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "README.md",
    "README_HOSTING.md",
    "requirements.txt",
    "requirements-dev.txt",
    "train_model.py",
    "app/streamlit_app.py",
    "data/airline_passengers_sample.csv",
    "models/airline_passenger_lstm.keras",
    "models/seasonal_growth_scaler.pkl",
    "models/model_metadata.json",
    "notebooks/airline_passenger_forecasting.ipynb",
    "tests/test_pipeline.py",
)


def main() -> None:
    missing = [item for item in REQUIRED_PATHS if not (PROJECT_ROOT / item).exists()]
    if missing:
        formatted = "\n".join(f"- {item}" for item in missing)
        raise SystemExit(f"Project validation failed. Missing paths:\n{formatted}")
    print(f"Project validation passed: {len(REQUIRED_PATHS)} required paths found.")


if __name__ == "__main__":
    main()
