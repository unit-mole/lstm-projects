from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def input_sequence_figure(sequence: np.ndarray):
    n_frames = len(sequence)
    fig, axes = plt.subplots(1, n_frames, figsize=(2.2 * n_frames, 2.6))
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.imshow(sequence[idx, :, :, 0], vmin=0, vmax=1)
        ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")
    fig.tight_layout()
    return fig


def comparison_figure(last_input: np.ndarray, actual: np.ndarray | None, predicted: np.ndarray):
    columns = 3 if actual is not None else 2
    fig, axes = plt.subplots(1, columns, figsize=(4 * columns, 3.6))
    axes = np.atleast_1d(axes)
    axes[0].imshow(last_input[:, :, 0], vmin=0, vmax=1)
    axes[0].set_title("Last input")
    offset = 1
    if actual is not None:
        axes[1].imshow(actual[:, :, 0], vmin=0, vmax=1)
        axes[1].set_title("Actual next frame")
        offset = 2
    axes[offset].imshow(predicted[:, :, 0], vmin=0, vmax=1)
    axes[offset].set_title("Predicted next frame")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def error_heatmap_figure(actual: np.ndarray, predicted: np.ndarray):
    error = np.abs(actual - predicted)[:, :, 0]
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(error)
    ax.set_title("Absolute error heatmap")
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
