from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def encode_source_text(cleaned_text: str, tokenizer: dict[str, Any], max_length: int):
    words = cleaned_text.split()
    word_index = tokenizer["word_index"]
    oov_id = int(tokenizer["oov_id"])
    retained_words = words[-max_length:]
    ids = [int(word_index.get(word, oov_id)) for word in retained_words]
    unknown_count = sum(token_id == oov_id for token_id in ids)
    oov_ratio = unknown_count / max(len(ids), 1)
    padded = np.zeros((1, max_length), dtype=np.int32)
    if ids:
        padded[0, :len(ids)] = np.asarray(ids, dtype=np.int32)
    return padded, retained_words, float(oov_ratio)
