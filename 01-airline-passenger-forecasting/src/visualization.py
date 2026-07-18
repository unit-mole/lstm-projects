from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_passenger_trend(frame: pd.DataFrame, path: str | Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(frame["Month"], frame["Passengers"], linewidth=2)
    plt.title("Monthly Airline Passenger Demand")
    plt.xlabel("Month")
    plt.ylabel("Passengers (thousands)")
    plt.grid(alpha=0.25)
    _save(path)


def plot_seasonal_pattern(frame: pd.DataFrame, path: str | Path) -> None:
    monthly = frame.groupby("MonthNumber", as_index=False)["Passengers"].mean()
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(10, 5))
    plt.plot(monthly["MonthNumber"], monthly["Passengers"], marker="o", linewidth=2)
    plt.xticks(range(1, 13), labels)
    plt.title("Average Seasonal Passenger Pattern")
    plt.xlabel("Month")
    plt.ylabel("Average passengers (thousands)")
    plt.grid(alpha=0.25)
    _save(path)


def plot_training_curve(history: dict[str, list[float]], path: str | Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(path)


def plot_actual_vs_predicted(predictions: pd.DataFrame, path: str | Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(predictions["Month"], predictions["Actual"], label="Actual", linewidth=2)
    plt.plot(predictions["Month"], predictions["Predicted"], label="LSTM prediction", linewidth=2, linestyle="--")
    plt.title("Actual vs Predicted Passengers — Test Period")
    plt.xlabel("Month")
    plt.ylabel("Passengers (thousands)")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(path)


def plot_residuals(predictions: pd.DataFrame, path: str | Path) -> None:
    plt.figure(figsize=(12, 4.5))
    plt.axhline(0, linewidth=1)
    plt.plot(predictions["Month"], predictions["Residual"], marker="o")
    plt.title("Forecast Residuals — Test Period")
    plt.xlabel("Month")
    plt.ylabel("Actual − predicted")
    plt.grid(alpha=0.25)
    _save(path)


def plot_forecast(frame: pd.DataFrame, forecast: pd.DataFrame, path: str | Path) -> None:
    plt.figure(figsize=(12, 5))
    history = frame.tail(60)
    plt.plot(history["Month"], history["Passengers"], label="Historical", linewidth=2)
    connector_x = [history["Month"].iloc[-1], forecast["Month"].iloc[0]]
    connector_y = [history["Passengers"].iloc[-1], forecast["Forecasted_Passengers"].iloc[0]]
    plt.plot(connector_x, connector_y, linestyle="--", alpha=0.6)
    plt.plot(forecast["Month"], forecast["Forecasted_Passengers"], label="Forecast", linewidth=2, linestyle="--")
    plt.title("Airline Passenger Forecast")
    plt.xlabel("Month")
    plt.ylabel("Passengers (thousands)")
    plt.legend()
    plt.grid(alpha=0.25)
    _save(path)


def plot_baseline_comparison(comparison: pd.DataFrame, path: str | Path) -> None:
    ordered = comparison.sort_values("rmse", ascending=True)
    x = np.arange(len(ordered))
    width = 0.36
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, ordered["mae"], width, label="MAE")
    plt.bar(x + width / 2, ordered["rmse"], width, label="RMSE")
    plt.xticks(x, ordered["Model"], rotation=25, ha="right")
    plt.title("Forecast Model and Baseline Comparison")
    plt.ylabel("Error (passengers, thousands)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _save(path)
