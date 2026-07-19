"""Matplotlib and in-memory image helpers for notebooks and Streamlit."""

from __future__ import annotations

from io import BytesIO

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_input_sequence(sequence: np.ndarray):
    frames = np.asarray(sequence)
    figure, axes = plt.subplots(1, len(frames), figsize=(2 * len(frames), 2.2))
    axes = np.atleast_1d(axes)
    for index, (axis, frame) in enumerate(zip(axes, frames), start=1):
        axis.imshow(frame[..., 0], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Frame {index}")
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_prediction_comparison(
    sequence: np.ndarray,
    predicted: np.ndarray,
    actual: np.ndarray | None = None,
):
    columns = 4 if actual is not None else 2
    figure, axes = plt.subplots(1, columns, figsize=(3.2 * columns, 3.2))
    axes = np.atleast_1d(axes)
    axes[0].imshow(sequence[-1, ..., 0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Last Input")
    cursor = 1
    if actual is not None:
        axes[cursor].imshow(actual[..., 0], cmap="gray", vmin=0, vmax=1)
        axes[cursor].set_title("Actual Next")
        cursor += 1
    axes[cursor].imshow(predicted[..., 0], cmap="gray", vmin=0, vmax=1)
    axes[cursor].set_title("Predicted Next")
    cursor += 1
    if actual is not None:
        error = np.abs(actual[..., 0] - predicted[..., 0])
        axes[cursor].imshow(error, cmap="magma", vmin=0, vmax=max(float(error.max()), 0.1))
        axes[cursor].set_title("Absolute Error")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    array = (np.clip(frame[..., 0], 0, 1) * 255).round().astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def frames_to_gif_bytes(frames: np.ndarray, duration: float = 0.35) -> bytes:
    rendered = []
    for frame in np.asarray(frames):
        gray = (np.clip(frame[..., 0], 0, 1) * 255).round().astype(np.uint8)
        rendered.append(np.stack([gray, gray, gray], axis=-1))
    buffer = BytesIO()
    imageio.mimsave(buffer, rendered, format="GIF", duration=duration, loop=0)
    return buffer.getvalue()
