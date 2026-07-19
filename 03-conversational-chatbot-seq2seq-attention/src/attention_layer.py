"""Attention documentation and optional Keras construction helper."""
from __future__ import annotations

ATTENTION_EXPLANATION = """
Additive attention compares each decoder state with every encoder output.
The resulting weights determine how strongly the decoder should use each
input position when predicting the next response token.
""".strip()

def build_keras_additive_attention(name: str = "attention_layer"):
    try:
        import keras
    except ImportError as exc:
        raise ImportError("Install the local training dependencies first.") from exc
    return keras.layers.AdditiveAttention(name=name)
