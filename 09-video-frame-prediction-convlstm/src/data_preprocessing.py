"""Frame normalization, resizing, and sample-data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Convert an image array to float32 pixels in the [0, 1] range."""
    arr = np.asarray(frame)
    if arr.size == 0:
        raise ValueError("Frame cannot be empty.")
    arr = arr.astype(np.float32)
    if float(arr.max()) > 1.0:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def resize_frame(
    frame: np.ndarray,
    target_size: tuple[int, int] = (32, 32),
    grayscale: bool = True,
) -> np.ndarray:
    """Resize one frame and return a channels-last float32 array."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    image = Image.fromarray(_to_uint8(arr))
    image = image.convert("L" if grayscale else "RGB")
    image = image.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    result = np.asarray(image, dtype=np.float32) / 255.0
    if grayscale:
        result = result[..., None]
    return result


def prepare_frame_sequence(
    frames: Iterable[np.ndarray],
    input_frames: int = 6,
    target_size: tuple[int, int] = (32, 32),
    grayscale: bool = True,
) -> np.ndarray:
    """Prepare a model-ready sequence with shape (time, height, width, channels)."""
    processed = [resize_frame(frame, target_size, grayscale) for frame in frames]
    if len(processed) < input_frames:
        raise ValueError(
            f"At least {input_frames} ordered frames are required; received {len(processed)}."
        )
    sequence = np.stack(processed[:input_frames]).astype(np.float32)
    expected_channels = 1 if grayscale else 3
    expected = (input_frames, target_size[0], target_size[1], expected_channels)
    if sequence.shape != expected:
        raise ValueError(f"Unexpected sequence shape {sequence.shape}; expected {expected}.")
    return sequence


def load_sample_sequences(path: str | Path) -> dict[str, np.ndarray]:
    """Load the safe synthetic sample dataset bundled with the project."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Sample dataset not found: {source}")
    with np.load(source, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    arr = normalize_frame(arr)
    return (arr * 255.0).round().astype(np.uint8)
