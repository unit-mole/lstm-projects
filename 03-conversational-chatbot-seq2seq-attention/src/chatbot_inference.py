from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .cloud_inference import NumpySeq2SeqAttention
from .config import FALLBACK_RESPONSE
from .retrieval_baseline import RetrievalResult, retrieve_response
from .text_preprocessing import validate_user_text
from .tokenizer_utils import encode_source_text, load_json

@dataclass
class ChatbotResult:
    cleaned_input: str
    response: str
    raw_model_response: str
    average_confidence: float
    token_confidences: list[float]
    generated_tokens: list[str]
    attention_weights: np.ndarray
    input_tokens: list[str]
    oov_ratio: float
    used_fallback: bool
    fallback_reason: str | None
    retrieval: RetrievalResult

class ChatbotService:
    def __init__(self, weights_path: str | Path, source_tokenizer_path: str | Path,
                 target_tokenizer_path: str | Path, tokenizer_meta_path: str | Path,
                 conversations: pd.DataFrame) -> None:
        self.model = NumpySeq2SeqAttention(weights_path)
        self.source_tokenizer = load_json(source_tokenizer_path)
        self.target_tokenizer = load_json(target_tokenizer_path)
        self.meta = load_json(tokenizer_meta_path)
        self.conversations = conversations.copy()
        self.target_index_word = {int(k): str(v) for k, v in self.target_tokenizer["index_word"].items()}

    def respond(self, text: object, confidence_threshold: float = 0.35,
                oov_threshold: float = 0.60) -> ChatbotResult:
        cleaned = validate_user_text(text)
        source_ids, input_tokens, oov_ratio = encode_source_text(
            cleaned, self.source_tokenizer, int(self.meta["max_src_len"])
        )
        generated = self.model.generate(
            source_ids, self.target_index_word,
            int(self.meta["start_token_id"]), int(self.meta["end_token_id"]),
            int(self.meta["max_tgt_len"])
        )
        retrieval = retrieve_response(cleaned, self.conversations)
        fallback_reason = None
        if not generated.response:
            fallback_reason = "The decoder did not produce a usable response."
        elif oov_ratio >= oov_threshold:
            fallback_reason = "Most input words are outside the model's small vocabulary."
        elif generated.average_confidence < confidence_threshold:
            fallback_reason = "The generated-token confidence was low."
        used_fallback = fallback_reason is not None
        return ChatbotResult(
            cleaned_input=cleaned,
            response=FALLBACK_RESPONSE if used_fallback else generated.response,
            raw_model_response=generated.response,
            average_confidence=generated.average_confidence,
            token_confidences=generated.token_confidences,
            generated_tokens=generated.generated_tokens,
            attention_weights=generated.attention_weights,
            input_tokens=input_tokens,
            oov_ratio=oov_ratio,
            used_fallback=used_fallback,
            fallback_reason=fallback_reason,
            retrieval=retrieval,
        )
