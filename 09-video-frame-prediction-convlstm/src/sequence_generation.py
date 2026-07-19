"""Synthetic data generation and sliding-window sequence construction."""

from __future__ import annotations

import numpy as np


def create_next_frame_sequences(
    frames: np.ndarray,
    input_frames: int,
    *,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create ordered input windows and next-frame targets from one frame stream."""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError("frames must have shape (time, height, width, channels)")
    if input_frames < 1 or stride < 1:
        raise ValueError("input_frames and stride must be positive")
    if len(arr) <= input_frames:
        raise ValueError("More frames than input_frames are required to create a target.")

    x, y = [], []
    for start in range(0, len(arr) - input_frames, stride):
        x.append(arr[start : start + input_frames])
        y.append(arr[start + input_frames])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def generate_moving_sequences(
    n_samples: int = 2500,
    seq_len: int = 6,
    image_size: int = 32,
    obj_size: int = 5,
    seed: int = 42,
    future_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the notebook's synthetic moving-square dataset."""
    if n_samples < 1 or seq_len < 1 or future_frames < 1:
        raise ValueError("n_samples, seq_len, and future_frames must be positive")
    if obj_size >= image_size:
        raise ValueError("obj_size must be smaller than image_size")

    rng = np.random.default_rng(seed)
    inputs, targets = [], []
    total_frames = seq_len + future_frames
    for _ in range(n_samples):
        frames = np.zeros((total_frames, image_size, image_size), dtype=np.float32)
        x = int(rng.integers(0, image_size - obj_size))
        y = int(rng.integers(0, image_size - obj_size))
        vx = int(rng.choice([-1, 1]) * rng.integers(1, 3))
        vy = int(rng.choice([-1, 1]) * rng.integers(1, 3))

        for time_step in range(total_frames):
            frames[time_step, y : y + obj_size, x : x + obj_size] = 1.0
            x += vx
            y += vy
            if x < 0 or x > image_size - obj_size:
                vx *= -1
                x = int(np.clip(x, 0, image_size - obj_size))
            if y < 0 or y > image_size - obj_size:
                vy *= -1
                y = int(np.clip(y, 0, image_size - obj_size))

        inputs.append(frames[:seq_len, ..., None])
        future = frames[seq_len:, ..., None]
        targets.append(future[0] if future_frames == 1 else future)
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)
