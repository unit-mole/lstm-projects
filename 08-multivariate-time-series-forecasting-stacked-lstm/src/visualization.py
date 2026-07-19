from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_actual_vs_predicted(actual, predicted, path: str | Path, timestamps=None) -> None:
    x_axis = timestamps if timestamps is not None else np.arange(len(actual))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(x_axis, actual, label="Actual")
    axis.plot(x_axis, predicted, label="Predicted")
    axis.set(title="Actual vs Predicted", xlabel="Time", ylabel="Target")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_residual_plot(actual, predicted, path: str | Path) -> None:
    residual_values = np.asarray(actual) - np.asarray(predicted)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(residual_values)
    axis.axhline(0, linestyle="--")
    axis.set(title="Residuals Over Time", xlabel="Observation", ylabel="Residual")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
