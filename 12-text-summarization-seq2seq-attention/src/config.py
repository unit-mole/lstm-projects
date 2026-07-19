"""Central project configuration and artifact paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGE_DIR = PROJECT_ROOT / "images"

TRAINING_MODEL_PATH = MODEL_DIR / "seq2seq_summarization.keras"
ENCODER_MODEL_PATH = MODEL_DIR / "encoder_summarization.keras"
DECODER_MODEL_PATH = MODEL_DIR / "decoder_summarization.keras"
SOURCE_TOKENIZER_PATH = MODEL_DIR / "source_tokenizer.json"
TARGET_TOKENIZER_PATH = MODEL_DIR / "target_tokenizer.json"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_METRICS_PATH = MODEL_DIR / "model_metrics.json"
SAMPLE_ARTICLES_PATH = DATA_DIR / "sample_articles.csv"

SEED = 42
MAX_SOURCE_LENGTH = 49
MAX_TARGET_LENGTH = 12
START_TOKEN = "sostok"
END_TOKEN = "eostok"
OOV_TOKEN = "<unk>"
MIN_INPUT_WORDS = 8
MAX_BATCH_ROWS = 100
