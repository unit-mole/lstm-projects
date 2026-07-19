from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_original_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    path: str | Path,
) -> None:
    time_axis = np.arange(len(np.asarray(original).reshape(-1)))
    plt.figure(figsize=(11, 5))
    plt.plot(time_axis, np.asarray(original).reshape(-1), label="Original signal")
    plt.plot(
        time_axis,
        np.asarray(reconstructed).reshape(-1),
        label="Reconstruction",
    )
    plt.xlabel("Timestep")
    plt.ylabel("Amplitude")
    plt.title("Original and Reconstructed ECG-Like Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_error_distribution(
    errors: np.ndarray,
    threshold: float,
    path: str | Path,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(np.asarray(errors, dtype=float), bins=40)
    plt.axvline(threshold, linestyle="--", label="Threshold")
    plt.xlabel("Reconstruction MAE")
    plt.ylabel("Sequences")
    plt.title("Reconstruction-Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def save_training_history(
    history: pd.DataFrame,
    path: str | Path,
) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["loss"], label="Training loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()
