"""Visualization functions used by the notebook and Streamlit application."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def create_attention_heatmap(
    attention_matrix: np.ndarray,
    source_tokens: list[str],
    generated_tokens: list[str],
):
    """Create a readable token-to-token additive-attention heatmap."""

    figure, axis = plt.subplots(
        figsize=(max(9, len(source_tokens) * 0.24), max(3.5, len(generated_tokens) * 0.5))
    )
    image = axis.imshow(attention_matrix, aspect="auto", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Attention weight")
    axis.set_xticks(range(len(source_tokens)))
    axis.set_xticklabels(source_tokens, rotation=75, ha="right")
    axis.set_yticks(range(len(generated_tokens)))
    axis.set_yticklabels(generated_tokens)
    axis.set_xlabel("Input tokens")
    axis.set_ylabel("Generated summary tokens")
    axis.set_title("Additive Attention Alignment")
    figure.tight_layout()
    return figure
