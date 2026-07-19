from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(min(4, torch.get_num_threads()))

from .model import FakeNewsLSTM, ModelConfig
from .text_preprocessing import TokenizerConfig, Vocabulary, encode_text


@dataclass(frozen=True)
class PredictionResult:
    predicted_label: str
    fake_probability: float
    confidence: float
    threshold: float
    interpretation: str
    raw_token_count: int
    used_token_count: int
    oov_ratio: float
    truncated: bool
    warning: str | None

    def to_dict(self) -> dict:
        return {
            "predicted_label": self.predicted_label,
            "fake_probability": self.fake_probability,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "interpretation": self.interpretation,
            "raw_token_count": self.raw_token_count,
            "used_token_count": self.used_token_count,
            "oov_ratio": self.oov_ratio,
            "truncated": self.truncated,
            "warning": self.warning,
        }


class FakeNewsPipeline:
    def __init__(
        self,
        model: FakeNewsLSTM,
        vocabulary: Vocabulary,
        tokenizer_config: TokenizerConfig,
        threshold: float,
        metadata: dict,
    ) -> None:
        self.model = model.eval()
        self.vocabulary = vocabulary
        self.tokenizer_config = tokenizer_config
        self.threshold = float(threshold)
        self.metadata = metadata

    @classmethod
    def load(cls, model_directory: str | Path) -> "FakeNewsPipeline":
        directory = Path(model_directory)
        checkpoint = torch.load(
            directory / "fake_news_lstm.pt",
            map_location="cpu",
            weights_only=True,
        )
        metadata = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
        tokenizer_payload = json.loads(
            (directory / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        vocabulary = Vocabulary.load(directory / "vocabulary.json")

        model_config_payload = checkpoint.get("model_config") or metadata["model_config"]
        model_config = ModelConfig(**model_config_payload)
        model = FakeNewsLSTM(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        tokenizer_config = TokenizerConfig(**tokenizer_payload)
        threshold = checkpoint.get("threshold", metadata["prediction_threshold"])
        return cls(model, vocabulary, tokenizer_config, threshold, metadata)

    def _interpret(self, probability: float, label: str) -> str:
        distance = abs(probability - self.threshold)
        if distance < 0.06:
            strength = "weak, threshold-adjacent"
        elif distance < 0.18:
            strength = "moderate"
        else:
            strength = "stronger"

        comparison = "fake-labeled" if label == "Fake" else "real-labeled"
        return (
            f"The model found {strength} similarity to language patterns in {comparison} "
            "LIAR training examples. This is not evidence that the claim is factually true or false."
        )

    def predict(self, text: object) -> PredictionResult:
        encoded, diagnostics = encode_text(text, self.vocabulary, self.tokenizer_config)
        token_tensor = torch.tensor([encoded], dtype=torch.long)
        with torch.no_grad():
            fake_probability = float(torch.sigmoid(self.model(token_tensor))[0].item())

        predicted_label = "Fake" if fake_probability >= self.threshold else "Real"
        confidence = max(fake_probability, 1.0 - fake_probability)

        warning = None
        if diagnostics["raw_token_count"] < 4:
            warning = "The input is very short; the prediction is unlikely to be reliable."
        elif diagnostics["oov_ratio"] > 0.40:
            warning = "Many words were unseen during training; this input may be out of domain."
        elif diagnostics["truncated"]:
            warning = "The input was truncated to the model's maximum sequence length."

        return PredictionResult(
            predicted_label=predicted_label,
            fake_probability=fake_probability,
            confidence=confidence,
            threshold=self.threshold,
            interpretation=self._interpret(fake_probability, predicted_label),
            raw_token_count=int(diagnostics["raw_token_count"]),
            used_token_count=int(diagnostics["used_token_count"]),
            oov_ratio=float(diagnostics["oov_ratio"]),
            truncated=bool(diagnostics["truncated"]),
            warning=warning,
        )

    def predict_batch(self, texts: Iterable[object]) -> pd.DataFrame:
        records = []
        for text in texts:
            result = self.predict(text)
            records.append({"text": "" if text is None else str(text), **result.to_dict()})
        return pd.DataFrame(records)
