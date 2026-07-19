from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

TOKEN_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)?|[!?]+|[%$]"
)

PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"
URL_TOKEN = "<URL>"
NUMBER_TOKEN = "<NUMBER>"
EXCLAMATION_TOKEN = "<EXCLAMATION>"
QUESTION_TOKEN = "<QUESTION>"
ALL_CAPS_TOKEN = "<ALL_CAPS_STYLE>"


@dataclass(frozen=True)
class TokenizerConfig:
    maximum_vocabulary_size: int = 15_000
    minimum_frequency: int = 2
    maximum_sequence_length: int = 48
    padding: str = "post"
    truncation: str = "post"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Vocabulary:
    index_to_token: list[str]

    def __post_init__(self) -> None:
        if not self.index_to_token or self.index_to_token[0] != PAD_TOKEN:
            raise ValueError(f"Vocabulary index 0 must be {PAD_TOKEN}.")
        if len(self.index_to_token) < 2 or self.index_to_token[1] != OOV_TOKEN:
            raise ValueError(f"Vocabulary index 1 must be {OOV_TOKEN}.")
        self.token_to_index = {token: index for index, token in enumerate(self.index_to_token)}

    def __len__(self) -> int:
        return len(self.index_to_token)

    @property
    def padding_index(self) -> int:
        return 0

    @property
    def oov_index(self) -> int:
        return 1

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.index_to_token, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        tokens = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(index_to_token=list(tokens))


def tokenize_text(text: object) -> list[str]:
    """Tokenize while preserving a small set of potentially useful style signals."""
    source = "" if text is None else str(text)
    raw_tokens = TOKEN_PATTERN.findall(source)
    normalized: list[str] = []

    uppercase_words = 0
    for token in raw_tokens:
        if token.startswith(("http://", "https://", "www.")):
            normalized.append(URL_TOKEN)
        elif token.startswith("!"):
            normalized.append(EXCLAMATION_TOKEN)
        elif token.startswith("?"):
            normalized.append(QUESTION_TOKEN)
        elif token[0].isdigit():
            normalized.append(NUMBER_TOKEN)
        elif token in {"%", "$"}:
            normalized.append(token)
        else:
            if token.isalpha() and len(token) > 2 and token.isupper():
                uppercase_words += 1
            normalized.append(token.lower())

    if uppercase_words >= 2:
        normalized.append(ALL_CAPS_TOKEN)
    return normalized


def build_vocabulary(
    texts: Iterable[object],
    config: TokenizerConfig,
) -> Vocabulary:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize_text(text))

    available_slots = max(config.maximum_vocabulary_size - 2, 0)
    learned_tokens = [
        token
        for token, frequency in counts.most_common(available_slots)
        if frequency >= config.minimum_frequency
    ]
    return Vocabulary(index_to_token=[PAD_TOKEN, OOV_TOKEN, *learned_tokens])


def encode_text(
    text: object,
    vocabulary: Vocabulary,
    config: TokenizerConfig,
) -> tuple[list[int], dict[str, float | int]]:
    tokens = tokenize_text(text)
    ids = [vocabulary.token_to_index.get(token, vocabulary.oov_index) for token in tokens]

    if config.truncation == "post":
        ids = ids[: config.maximum_sequence_length]
    elif config.truncation == "pre":
        ids = ids[-config.maximum_sequence_length :]
    else:
        raise ValueError("truncation must be 'pre' or 'post'.")

    unpadded_length = len(ids)
    padding_needed = max(config.maximum_sequence_length - unpadded_length, 0)
    padding = [vocabulary.padding_index] * padding_needed
    if config.padding == "post":
        padded = ids + padding
    elif config.padding == "pre":
        padded = padding + ids
    else:
        raise ValueError("padding must be 'pre' or 'post'.")

    token_count = max(len(tokens), 1)
    oov_count = sum(index == vocabulary.oov_index for index in ids)
    diagnostics = {
        "raw_token_count": len(tokens),
        "used_token_count": unpadded_length,
        "oov_count": oov_count,
        "oov_ratio": float(oov_count / token_count),
        "truncated": int(len(tokens) > config.maximum_sequence_length),
    }
    return padded, diagnostics


def encode_texts(
    texts: Sequence[object],
    vocabulary: Vocabulary,
    config: TokenizerConfig,
) -> np.ndarray:
    encoded = [encode_text(text, vocabulary, config)[0] for text in texts]
    return np.asarray(encoded, dtype=np.int64)
