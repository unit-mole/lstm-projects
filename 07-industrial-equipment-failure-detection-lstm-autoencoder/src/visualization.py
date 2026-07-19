from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def sensor_trend_figure(
    frame: pd.DataFrame,
    time_col: str,
    sensor_cols: list[str],
    window_start=None,
    window_end=None,
):
    long = frame[[time_col, *sensor_cols]].melt(
        id_vars=time_col, var_name="sensor", value_name="reading"
    )
    figure = px.line(long, x=time_col, y="reading", color="sensor", title="Sensor trends")
    if window_start is not None and window_end is not None:
        figure.add_vrect(x0=window_start, x1=window_end, opacity=0.12, line_width=0)
    figure.update_layout(legend_title_text="Sensor", hovermode="x unified")
    return figure


def reconstruction_figure(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sensor_cols: list[str],
    selected_sensors: list[str],
):
    figure = go.Figure()
    for sensor in selected_sensors:
        index = sensor_cols.index(sensor)
        figure.add_trace(go.Scatter(y=original[:, index], name=f"{sensor} — original"))
        figure.add_trace(go.Scatter(
            y=reconstructed[:, index], name=f"{sensor} — reconstructed", line={"dash": "dash"}
        ))
    figure.update_layout(
        title="Original vs reconstructed sequence (scaled space)",
        xaxis_title="Time step within selected window",
        yaxis_title="Standardized sensor reading",
        hovermode="x unified",
    )
    return figure


def error_timeline_figure(predictions: pd.DataFrame, threshold: float):
    figure = px.line(
        predictions,
        x="window_end",
        y="reconstruction_error",
        markers=True,
        color="health_status",
        title="Equipment health timeline",
        hover_data=["anomaly_score", "predicted_anomaly"],
    )
    figure.add_hline(y=threshold, line_dash="dash", annotation_text="Anomaly threshold")
    figure.add_hline(y=1.5 * threshold, line_dash="dot", annotation_text="High-risk band")
    return figure


def sensor_contribution_figure(sensor_cols: list[str], errors: np.ndarray):
    data = pd.DataFrame({"sensor": sensor_cols, "mean_absolute_error": errors})
    data = data.sort_values("mean_absolute_error", ascending=True)
    return px.bar(
        data, x="mean_absolute_error", y="sensor", orientation="h",
        title="Sensor contribution to reconstruction error",
    )
