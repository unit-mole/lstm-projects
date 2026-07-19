from __future__ import annotations

import numpy as np


def coordinate_grid(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")
    return np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))


def weather_blob(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    intensity: float,
    spread: float,
    time_index: int,
    noise: np.ndarray,
) -> np.ndarray:
    gaussian = np.exp(-(((xx - center_x) ** 2) + ((yy - center_y) ** 2)) / (2 * spread**2))
    wave = 0.15 * np.sin(2 * np.pi * (xx + yy + time_index / 8.0))
    return np.clip(intensity * gaussian + wave + noise, 0.0, 1.0).astype("float32")
