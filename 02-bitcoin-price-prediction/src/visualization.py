from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def save_line_plot(
    data: pd.DataFrame,
    x: str,
    y_columns: list[str],
    title: str,
    ylabel: str,
    output_path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for column in y_columns:
        ax.plot(data[x], data[column], label=column)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
