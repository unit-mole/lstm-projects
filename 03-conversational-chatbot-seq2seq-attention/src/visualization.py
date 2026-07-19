from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def save_training_curves(history: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9,5))
    plt.plot(history["epoch"], history["loss"], label="Training loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Sparse categorical cross-entropy")
    plt.title("Seq2Seq Attention Training and Validation Loss"); plt.legend()
    plt.tight_layout(); plt.savefig(output_dir/"training_curve.png", dpi=160); plt.close()
    plt.figure(figsize=(9,5))
    plt.plot(history["epoch"], history["accuracy"], label="Training accuracy")
    plt.plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Token accuracy")
    plt.title("Seq2Seq Attention Token Accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(output_dir/"token_accuracy_curve.png", dpi=160); plt.close()

def save_attention_heatmap(attention: np.ndarray, input_tokens: list[str],
                           output_tokens: list[str], path: str | Path) -> None:
    matrix = np.asarray(attention, dtype=float)
    if matrix.size == 0: return
    matrix = matrix[:, :max(len(input_tokens),1)]
    sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)
    plt.figure(figsize=(8,5)); plt.imshow(matrix, aspect="auto")
    plt.xticks(range(len(input_tokens)), input_tokens, rotation=30, ha="right")
    plt.yticks(range(len(output_tokens)), output_tokens)
    plt.xlabel("Encoder input tokens"); plt.ylabel("Generated response tokens")
    plt.title("Additive Attention Weights"); plt.colorbar(label="Normalized attention")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
