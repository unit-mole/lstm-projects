"""Shared configuration and filesystem paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

MODEL_PATH = MODEL_DIR / "convlstm_video_prediction.keras"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
SAMPLE_DATA_PATH = DATA_DIR / "sample_sequences.npz"
MULTISTEP_SAMPLE_PATH = DATA_DIR / "sample_multistep_sequence.npz"

DEFAULT_INPUT_FRAMES = 6
DEFAULT_HEIGHT = 32
DEFAULT_WIDTH = 32
DEFAULT_CHANNELS = 1
DEFAULT_SEED = 42
