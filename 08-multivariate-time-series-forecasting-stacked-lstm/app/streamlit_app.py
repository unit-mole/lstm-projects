from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import prepare_time_series
from src.feature_engineering import add_calendar_features
from src.inference_pipeline import (
    ScalerArtifacts,
    load_keras_model,
    recursive_forecast,
    seasonal_future_exogenous,
)

st.set_page_config(
    page_title="Stacked LSTM Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
    .hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#0f172a,#1e3a8a); color:white; margin-bottom:1rem;}
    .hero h1 {margin:0; font-size:2.1rem;}
    .hero p {margin:.55rem 0 0; opacity:.9;}
    .metric-card {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:1rem; min-height:110px;}
    .small-note {font-size:.88rem; opacity:.78;}
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = PROJECT_ROOT / "models" / "stacked_lstm_energy.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scalers.json"
METRICS_PATH = PROJECT_ROOT / "outputs" / "model_metrics.json"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_multivariate_timeseries.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "test_predictions.csv"
BASELINE_PATH = PROJECT_ROOT / "outputs" / "baseline_comparison.csv"


@st.cache_data
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model_cached():
    return load_keras_model(MODEL_PATH)


metadata = load_json(METADATA_PATH)
metrics = load_json(METRICS_PATH)
scalers = ScalerArtifacts.from_json(SCALER_PATH)

st.markdown(
    """
    <div class="hero">
      <h1>Multivariate Time Series Forecasting using Stacked LSTM</h1>
      <p>Forecast hourly energy demand from historical load, weather variables and cyclical calendar signals using a pre-trained 3-layer LSTM.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Data source")
    source = st.radio("Choose input", ["Portfolio sample", "Upload CSV"], index=0)
    uploaded = st.file_uploader("Upload a multivariate time-series CSV", type=["csv"], disabled=source == "Portfolio sample")
    st.divider()
    st.subheader("Model contract")
    st.caption("The deployed artifact is fixed to a 24-hour input window and these eight features:")
    st.code("\n".join(scalers.feature_columns), language=None)
    st.caption("Target: energy_load • Frequency: hourly • Output: next-hour forecast")

if source == "Upload CSV" and uploaded is not None:
    raw = pd.read_csv(uploaded)
else:
    raw = load_csv(SAMPLE_PATH)

if source == "Upload CSV" and uploaded is None:
    st.info("Upload a CSV to replace the portfolio sample. The sample remains visible until a file is supplied.")

# Flexible column mapping for uploaded files while preserving the pre-trained model contract.
st.subheader("Column mapping")
columns = list(raw.columns)
def default_index(name: str, fallback: int = 0) -> int:
    return columns.index(name) if name in columns else min(fallback, len(columns) - 1)

mapping_columns = st.columns(4)
with mapping_columns[0]:
    timestamp_column = st.selectbox("Timestamp column", columns, index=default_index("timestamp"))
with mapping_columns[1]:
    target_column = st.selectbox("Target column", columns, index=default_index("energy_load"))
with mapping_columns[2]:
    temperature_column = st.selectbox("Temperature feature", columns, index=default_index("temperature"))
with mapping_columns[3]:
    humidity_column = st.selectbox("Humidity feature", columns, index=default_index("humidity"))

standardized = raw.rename(columns={
    timestamp_column: "timestamp",
    target_column: "energy_load",
    temperature_column: "temperature",
    humidity_column: "humidity",
})
try:
    prepared, quality = prepare_time_series(
        standardized,
        timestamp_column="timestamp",
        required_numeric_columns=("energy_load", "temperature", "humidity"),
    )
    prepared = add_calendar_features(prepared)
except Exception as error:
    st.error(f"The selected data cannot be prepared: {error}")
    st.stop()

model_metrics = metrics["stacked_lstm_test"]
card_columns = st.columns(5)
card_values = [
    ("Test MAE", f"{model_metrics['mae']:.2f}"),
    ("Test RMSE", f"{model_metrics['rmse']:.2f}"),
    ("Test MAPE", f"{model_metrics['mape_pct']:.2f}%"),
    ("Test R²", f"{model_metrics['r2']:.3f}"),
    ("Input tensor", f"24 × {len(scalers.feature_columns)}"),
]
for container, (label, value) in zip(card_columns, card_values):
    container.metric(label, value)

tab_data, tab_performance, tab_forecast, tab_details = st.tabs([
    "Data Explorer", "Model Performance", "Forecast Lab", "Project Details"
])

with tab_data:
    st.subheader("Data quality and preview")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Rows", f"{quality.rows:,}")
    summary_columns[1].metric("Columns", quality.columns)
    summary_columns[2].metric("Duplicates removed", quality.duplicate_timestamps_removed)
    summary_columns[3].metric("Missing after cleaning", sum(quality.missing_values_after.values()))
    st.dataframe(prepared.head(25), use_container_width=True, hide_index=True)

    trend_columns = st.columns(2)
    with trend_columns[0]:
        target_fig = px.line(prepared.tail(min(720, len(prepared))), x="timestamp", y="energy_load", title="Target trend")
        st.plotly_chart(target_fig, use_container_width=True)
    with trend_columns[1]:
        weather_long = prepared.tail(min(720, len(prepared))).melt(
            id_vars="timestamp", value_vars=["temperature", "humidity"], var_name="feature", value_name="value"
        )
        feature_fig = px.line(weather_long, x="timestamp", y="value", color="feature", title="Weather features")
        st.plotly_chart(feature_fig, use_container_width=True)

    numeric = prepared.select_dtypes(include=np.number)
    correlation = numeric.corr()
    heatmap = px.imshow(correlation, text_auto=".2f", aspect="auto", title="Correlation heatmap")
    st.plotly_chart(heatmap, use_container_width=True)

with tab_performance:
    st.subheader("Future-period test performance")
    prediction_data = load_csv(PREDICTIONS_PATH)
    prediction_data["timestamp"] = pd.to_datetime(prediction_data["timestamp"])
    displayed = prediction_data.head(500)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=displayed["timestamp"], y=displayed["actual_energy_load"], name="Actual"))
    figure.add_trace(go.Scatter(x=displayed["timestamp"], y=displayed["stacked_lstm_prediction"], name="Stacked LSTM"))
    figure.update_layout(title="Actual vs predicted — first 500 test observations", xaxis_title="Timestamp", yaxis_title="Energy load")
    st.plotly_chart(figure, use_container_width=True)

    baseline = load_csv(BASELINE_PATH)
    st.dataframe(baseline.style.format({"mae": "{:.3f}", "rmse": "{:.3f}", "mape_pct": "{:.2f}%", "r2": "{:.3f}"}), use_container_width=True, hide_index=True)
    comparison = baseline.melt(id_vars="model", value_vars=["mae", "rmse"], var_name="metric", value_name="value")
    st.plotly_chart(px.bar(comparison, x="model", y="value", color="metric", barmode="group", title="Baseline comparison"), use_container_width=True)

    residual_fig = px.scatter(
        prediction_data.head(1000), x="timestamp", y="residual", title="Residual behavior — first 1,000 test observations"
    )
    residual_fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(residual_fig, use_container_width=True)
    st.success(
        f"The Stacked LSTM reduced test MAE by {metrics['improvement_vs_naive_test_pct']['mae_reduction']:.1f}% "
        f"and RMSE by {metrics['improvement_vs_naive_test_pct']['rmse_reduction']:.1f}% versus the previous-value baseline."
    )

with tab_forecast:
    st.subheader("Interactive future forecast")
    st.caption(
        "The supplied artifact is a one-step model. This demo creates a multi-hour forecast recursively, "
        "using either uploaded future temperature/humidity values or a transparent 24-hour seasonal-naive assumption."
    )
    forecast_columns = st.columns([1, 2])
    with forecast_columns[0]:
        horizon = st.slider("Forecast horizon (hours)", 1, 24, 7)
        future_file = st.file_uploader(
            "Optional future exogenous CSV (timestamp, temperature, humidity)", type=["csv"], key="future_exogenous"
        )
    with forecast_columns[1]:
        st.markdown(
            "**Required history:** at least 24 chronologically ordered rows containing timestamp, target, temperature and humidity."
        )
        st.markdown(
            "**Business use:** short-horizon load forecasts can support capacity planning, staffing, resource allocation and operational decision-making."
        )

    if st.button("Generate forecast", type="primary", use_container_width=True):
        if len(prepared) < scalers.sequence_length:
            st.error(f"At least {scalers.sequence_length} historical rows are required.")
        else:
            if future_file is not None:
                future = pd.read_csv(future_file).head(horizon)
            else:
                future = seasonal_future_exogenous(prepared, horizon)
            if len(future) < horizon:
                st.error(f"Future exogenous input contains {len(future)} rows, but horizon is {horizon}.")
            else:
                try:
                    with st.spinner("Loading the pre-trained model and generating forecasts..."):
                        model = load_model_cached()
                        forecast = recursive_forecast(model, prepared, future.head(horizon), scalers)
                except Exception as error:
                    st.error(f"Forecasting could not run: {error}")
                else:
                    forecast_value = float(forecast["forecasted_energy_load"].iloc[-1])
                    forecast_mean = float(forecast["forecasted_energy_load"].mean())
                    result_columns = st.columns(3)
                    result_columns[0].metric("Final forecast", f"{forecast_value:,.2f}")
                    result_columns[1].metric("Horizon average", f"{forecast_mean:,.2f}")
                    result_columns[2].metric("Forecast horizon", f"{horizon} hour{'s' if horizon != 1 else ''}")

                    history_tail = prepared.tail(72)[["timestamp", "energy_load"]]
                    forecast_fig = go.Figure()
                    forecast_fig.add_trace(go.Scatter(x=history_tail["timestamp"], y=history_tail["energy_load"], name="Historical"))
                    forecast_fig.add_trace(go.Scatter(x=forecast["timestamp"], y=forecast["forecasted_energy_load"], name="Forecast", mode="lines+markers"))
                    forecast_fig.update_layout(title="Historical context and future forecast", xaxis_title="Timestamp", yaxis_title="Energy load")
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    st.dataframe(forecast, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download forecast CSV",
                        forecast.to_csv(index=False).encode("utf-8"),
                        file_name="stacked_lstm_forecast.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

with tab_details:
    st.subheader("Architecture and project scope")
    st.code("\n↓\n".join(metadata["model_architecture"]), language=None)
    st.markdown(
        "**Multivariate setup:** each sample contains the prior 24 hourly observations across eight features. "
        "The output is the next-hour energy-load value. The target's own history is included alongside temperature, humidity and cyclical calendar variables."
    )
    st.markdown("**Limitations**")
    for limitation in metadata["limitations"]:
        st.write(f"• {limitation}")
    st.info("GitHub repository link: replace this placeholder after pushing the project to your `lstm-projects` monorepo.")
