from __future__ import annotations

import numpy as np

from src.grid_generation import coordinate_grid, weather_blob


def generate_weather_sequences(
    n_samples: int = 2200,
    input_frames: int = 6,
    future_frames: int = 1,
    grid_size: int = 24,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent moving synthetic weather systems in [0, 1]."""
    if n_samples < 1 or input_frames < 1 or future_frames < 1:
        raise ValueError("n_samples, input_frames, and future_frames must be positive")

    rng = np.random.default_rng(seed)
    xx, yy = coordinate_grid(grid_size)
    X, future = [], []

    for _ in range(n_samples):
        cx, cy = rng.uniform(-0.4, 0.4, size=2)
        vx, vy = rng.uniform(-0.12, 0.12, size=2)
        intensity = rng.uniform(0.6, 1.2)
        spread = rng.uniform(0.18, 0.35)
        frames = []

        for time_index in range(input_frames + future_frames):
            noise = rng.normal(0, 0.02, size=xx.shape)
            frame = weather_blob(xx, yy, cx, cy, intensity, spread, time_index, noise)
            frames.append(frame[..., None])
            cx += vx
            cy += vy
            if cx < -0.8 or cx > 0.8:
                vx *= -1
                cx = float(np.clip(cx, -0.8, 0.8))
            if cy < -0.8 or cy > 0.8:
                vy *= -1
                cy = float(np.clip(cy, -0.8, 0.8))

        frames_array = np.asarray(frames, dtype="float32")
        X.append(frames_array[:input_frames])
        future.append(frames_array[input_frames:])

    X_array = np.asarray(X, dtype="float32")
    future_array = np.asarray(future, dtype="float32")
    return X_array, future_array
