from __future__ import annotations
import numpy as np

def encode_texts(texts: list[str], word_index: dict[str, int], max_length: int, oov_id: int):
    output = np.zeros((len(texts), max_length), dtype=np.int32)
    for row_index, text in enumerate(texts):
        ids = [int(word_index.get(word, oov_id)) for word in text.split()]
        ids = ids[-max_length:]
        output[row_index, :len(ids)] = np.asarray(ids, dtype=np.int32)
    return output

def build_teacher_forcing_arrays(target_texts: list[str], word_index: dict[str, int],
                                 max_target_length: int, oov_id: int):
    full = encode_texts(target_texts, word_index, max_target_length, oov_id)
    return full[:, :-1], full[:, 1:]
