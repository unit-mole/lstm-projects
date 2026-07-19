from __future__ import annotations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "images"

SAMPLE_DATA_PATH = DATA_DIR / "sample_conversations.csv"
SAMPLE_PROMPTS_PATH = DATA_DIR / "sample_prompts.json"
WEIGHTS_PATH = MODELS_DIR / "seq2seq_attention_weights.npz"
SOURCE_TOKENIZER_PATH = MODELS_DIR / "source_tokenizer.json"
TARGET_TOKENIZER_PATH = MODELS_DIR / "target_tokenizer.json"
TOKENIZER_META_PATH = MODELS_DIR / "tokenizer_meta.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

FALLBACK_RESPONSE = (
    "I am not fully sure how to answer that yet. "
    "This chatbot is a portfolio demo trained on a limited dataset."
)
RESPONSIBLE_USE_NOTE = (
    "Educational portfolio demonstration only. Responses may be inaccurate, repetitive, "
    "biased, incomplete, or nonsensical. Do not enter private or sensitive information, "
    "and do not use this chatbot for medical, legal, financial, safety-critical, or "
    "production customer-support decisions."
)
