"""Reusable Matplotlib visualizations for notebooks and reports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def traffic_trend_figure(frame: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(frame["timestamp"], frame["congestion_index"])
    axis.set_title("Traffic Congestion Index Over Time")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Congestion index")
    figure.tight_layout()
    return figure


def actual_vs_predicted_figure(predictions: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(
        predictions["timestamp"],
        predictions["actual_congestion_index"],
        label="Actual",
    )
    axis.plot(
        predictions["timestamp"],
        predictions["predicted_congestion_index"],
        label="Predicted",
    )
    axis.set_title("Actual vs Predicted Traffic Congestion")
    axis.legend()
    figure.tight_layout()
    return figure


def residual_figure(predictions: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(predictions["timestamp"], predictions["residual"])
    axis.axhline(0)
    axis.set_title("Forecast Residuals Over Time")
    axis.set_ylabel("Actual - predicted")
    figure.tight_layout()
    return figure
