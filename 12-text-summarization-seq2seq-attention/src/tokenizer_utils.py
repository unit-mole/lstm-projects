"""Portable tokenizer utilities independent of TensorFlow pickle objects."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PortableTokenizer:
    """Small whitespace tokenizer compatible with this project's saved vocabularies."""

    word_index: dict[str, int]
    index_word: dict[int, str]
    oov_token: str = "<unk>"

    @property
    def vocab_size(self) -> int:
        return len(self.word_index) + 1

    @property
    def oov_id(self) -> int:
        return self.word_index[self.oov_token]

    def text_to_sequence(self, text: str) -> list[int]:
        return [self.word_index.get(token, self.oov_id) for token in text.split()]

    def texts_to_sequences(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.text_to_sequence(text) for text in texts]

    def tokens_to_text(
        self,
        token_ids: Iterable[int],
        excluded_tokens: set[str] | None = None,
    ) -> str:
        excluded = excluded_tokens or set()
        words = [
            self.index_word.get(int(token_id), "")
            for token_id in token_ids
            if int(token_id) != 0
        ]
        return " ".join(word for word in words if word and word not in excluded)

    def oov_ratio(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0.0
        unknown = sum(word not in self.word_index for word in words)
        return unknown / len(words)

    @classmethod
    def load(cls, path: str | Path) -> "PortableTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            word_index={str(key): int(value) for key, value in payload["word_index"].items()},
            index_word={int(key): str(value) for key, value in payload["index_word"].items()},
            oov_token=str(payload.get("config", {}).get("oov_token", "<unk>")),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "class_name": "PortableWhitespaceTokenizer",
            "config": {"oov_token": self.oov_token, "lower": True, "split": " "},
            "word_index": self.word_index,
            "index_word": {str(key): value for key, value in self.index_word.items()},
            "vocab_size": self.vocab_size,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fit_portable_tokenizer(
    texts: Iterable[str],
    oov_token: str = "<unk>",
) -> PortableTokenizer:
    """Fit a deterministic frequency tokenizer with stable tie ordering.

    Stable frequency sorting reproduces the word-index ordering used by the
    legacy Keras Tokenizer in the supplied notebook.
    """

    counts: OrderedDict[str, int] = OrderedDict()
    for text in texts:
        for word in str(text).split():
            counts[word] = counts.get(word, 0) + 1

    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    word_index = {oov_token: 1}
    for word, _ in ordered:
        if word != oov_token and word not in word_index:
            word_index[word] = len(word_index) + 1
    return PortableTokenizer(
        word_index=word_index,
        index_word={index: word for word, index in word_index.items()},
        oov_token=oov_token,
    )
