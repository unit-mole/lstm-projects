from __future__ import annotations
import pandas as pd
from src.chatbot_inference import ChatbotService
from src.config import (
    SAMPLE_DATA_PATH, SOURCE_TOKENIZER_PATH, TARGET_TOKENIZER_PATH,
    TOKENIZER_META_PATH, WEIGHTS_PATH,
)

def build_service():
    return ChatbotService(
        WEIGHTS_PATH, SOURCE_TOKENIZER_PATH, TARGET_TOKENIZER_PATH,
        TOKENIZER_META_PATH, pd.read_csv(SAMPLE_DATA_PATH)
    )

def test_canonical_prompt_generates_expected_response():
    result = build_service().respond("hello")
    assert result.raw_model_response == "hi there"
    assert result.response == "hi there"
    assert not result.used_fallback
    assert result.average_confidence > 0.90

def test_attention_shape_matches_tokens():
    result = build_service().respond("what should i do next")
    assert result.attention_weights.shape[0] == len(result.generated_tokens)
    assert result.attention_weights.shape[1] == 5

def test_out_of_domain_input_uses_responsible_fallback():
    result = build_service().respond("quantum astrophysics blockchain regulation")
    assert result.used_fallback
    assert "portfolio demo" in result.response.lower()
