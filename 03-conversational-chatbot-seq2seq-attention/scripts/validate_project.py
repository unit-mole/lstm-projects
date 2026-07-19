from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot_inference import ChatbotService
from src.config import (
    SAMPLE_DATA_PATH, SOURCE_TOKENIZER_PATH, TARGET_TOKENIZER_PATH,
    TOKENIZER_META_PATH, WEIGHTS_PATH,
)

REQUIRED_FILES = [
    "README.md","README_HOSTING.md","app/streamlit_app.py","app/requirements.txt",
    "data/sample_conversations.csv","data/sample_prompts.json",
    "models/seq2seq_chatbot.keras","models/encoder_chatbot.keras","models/decoder_chatbot.keras",
    "models/seq2seq_attention_weights.npz","models/source_tokenizer.json",
    "models/target_tokenizer.json","models/tokenizer_meta.json","models/model_metadata.json",
    "outputs/training_curve.png","outputs/token_accuracy_curve.png",
    "outputs/attention_visualization.png","outputs/sample_chat_responses.csv",
    "notebooks/conversational_chatbot_seq2seq_attention.ipynb",
]

def main():
    missing = [x for x in REQUIRED_FILES if not (PROJECT_ROOT/x).exists()]
    if missing:
        raise FileNotFoundError(f"Required files are missing: {missing}")
    conversations = pd.read_csv(SAMPLE_DATA_PATH)
    service = ChatbotService(
        WEIGHTS_PATH, SOURCE_TOKENIZER_PATH, TARGET_TOKENIZER_PATH,
        TOKENIZER_META_PATH, conversations
    )
    expected = {
        "hello":"hi there",
        "how are you":"i am doing well",
        "what should i do next":"you should review the latest update",
    }
    generated = {prompt:service.respond(prompt).raw_model_response for prompt in expected}
    if generated != expected:
        raise RuntimeError(f"Canonical response validation failed: {generated}")
    attention_result = service.respond("can you summarize this")
    checks = {
        "required_files":"passed",
        "sample_pairs":int(len(conversations)),
        "canonical_responses":"passed",
        "attention_rows":int(attention_result.attention_weights.shape[0]),
        "attention_columns":int(attention_result.attention_weights.shape[1]),
        "confidence_finite":bool(np.isfinite(attention_result.average_confidence)),
        "cloud_runtime":"NumPy encoder-decoder LSTM with additive attention",
    }
    print(json.dumps(checks,indent=2))
    print("Seq2Seq attention chatbot project validation passed.")

if __name__ == "__main__":
    main()
