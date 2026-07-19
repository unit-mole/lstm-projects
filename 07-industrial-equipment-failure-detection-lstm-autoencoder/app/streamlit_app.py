from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SAMPLE_DATA, PATHS
from src.feature_engineering import sensor_summary
from src.inference_pipeline import PredictiveMaintenancePipeline
from src.visualization import (
    error_timeline_figure,
    reconstruction_figure,
    sensor_contribution_figure,
    sensor_trend_figure,
)

st.set_page_config(
    page_title="Industrial Equipment Failure Detection",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
      .safety-box {border-left: 5px solid #d97706; padding: 0.85rem 1rem;
                   background: rgba(217,119,6,0.08); border-radius: 0.35rem;}
      .small-note {font-size: 0.88rem; opacity: 0.82;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained LSTM Autoencoder artifacts...")
def load_pipeline() -> PredictiveMaintenancePipeline:
    return PredictiveMaintenancePipeline.from_artifacts(PATHS.model_dir)


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_SAMPLE_DATA)


pipeline = load_pipeline()
metadata = pipeline.metadata
sensor_cols = list(metadata["sensor_cols"])
threshold = float(metadata["threshold"])

st.title("⚙️ Industrial Equipment Failure Detection")
st.caption(
    "LSTM Autoencoder · multivariate time-series anomaly detection · predictive maintenance demo"
)
st.markdown(
    """
    <div class="safety-box"><strong>Responsible-use notice:</strong> This portfolio project is
    for educational demonstration only. It must not be used as the sole basis for industrial
    maintenance, safety, production, or operational decisions. Real deployments require validated
    sensors, maintenance history, domain expertise, operational context, and qualified human review.
    False positives and false negatives are possible.</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo controls")
    source_choice = st.radio("Data source", ["Use sample data", "Upload CSV"])
    if source_choice == "Use sample data":
        raw_frame = load_sample()
        st.success("Safe synthetic demonstration data loaded.")
    else:
        uploaded = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV or switch to the sample dataset.")
            st.stop()
        raw_frame = pd.read_csv(uploaded)

    missing_model_columns = [c for c in [metadata["unit_id_col"], metadata["time_col"], *sensor_cols] if c not in raw_frame.columns]
    if missing_model_columns:
        st.error("The uploaded file is missing required columns: " + ", ".join(missing_model_columns))
        st.stop()

    units = sorted(raw_frame[metadata["unit_id_col"]].dropna().unique().tolist())
    default_index = units.index(105) if 105 in units else 0
    selected_unit = st.selectbox("Equipment / unit ID", units, index=default_index)
    step_size = st.select_slider("Window step size", options=[1, 2, 5, 10], value=1)
    st.caption(f"Model window: {metadata['seq_len']} rows · {len(sensor_cols)} sensors")
    st.caption(f"Inference backend: {pipeline.backend_name}")

try:
    result = pipeline.score_dataframe(raw_frame, selected_unit=selected_unit, step_size=step_size)
except Exception as exc:
    st.error(f"Unable to analyze the selected data: {exc}")
    st.stop()

prediction_table = result.predictions
sequence_count = len(prediction_table)
with st.sidebar:
    selected_sequence = st.slider(
        "Sequence to inspect", min_value=0, max_value=sequence_count - 1,
        value=min(sequence_count - 1, max(0, sequence_count // 2)),
    )

selected = prediction_table.iloc[selected_sequence]
error = float(selected["reconstruction_error"])
score = float(selected["anomaly_score"])
status = str(selected["health_status"])

st.subheader("Selected equipment assessment")
metric_cols = st.columns(4)
metric_cols[0].metric("Equipment health status", status)
metric_cols[1].metric("Reconstruction error", f"{error:.4f}")
metric_cols[2].metric("Anomaly threshold", f"{threshold:.4f}")
metric_cols[3].metric("Anomaly score", f"{score:.2f}× threshold")

st.info(str(selected["risk_interpretation"]))

start_pos = int(result.batch.start_positions[selected_sequence])
end_pos = int(result.batch.end_positions[selected_sequence])
time_col = metadata["time_col"]
window_start = result.clean_frame.iloc[start_pos][time_col]
window_end = result.clean_frame.iloc[end_pos][time_col]

trend_sensors = st.multiselect(
    "Sensors shown in the full equipment trend",
    sensor_cols,
    default=sensor_cols[:4],
)
if trend_sensors:
    st.plotly_chart(
        sensor_trend_figure(
            result.clean_frame, time_col, trend_sensors,
            window_start=window_start, window_end=window_end,
        ),
        use_container_width=True,
    )

left, right = st.columns(2)
with left:
    reconstruction_sensors = st.multiselect(
        "Sensors shown in reconstruction comparison",
        sensor_cols,
        default=sensor_cols[:2],
        key="reconstruction_sensors",
    )
    if reconstruction_sensors:
        st.plotly_chart(
            reconstruction_figure(
                result.batch.sequences[selected_sequence],
                result.reconstructions[selected_sequence],
                sensor_cols,
                reconstruction_sensors,
            ),
            use_container_width=True,
        )
with right:
    st.plotly_chart(
        sensor_contribution_figure(sensor_cols, result.sensor_errors[selected_sequence]),
        use_container_width=True,
    )

st.plotly_chart(error_timeline_figure(prediction_table, threshold), use_container_width=True)

with st.expander("Data preview and sensor-quality summary"):
    st.write("Cleaned selected-equipment data")
    st.dataframe(result.clean_frame.head(30), use_container_width=True)
    st.write("Sensor summary")
    st.dataframe(sensor_summary(result.clean_frame, sensor_cols), use_container_width=True)

with st.expander("Model details and portfolio metrics"):
    st.markdown(
        f"""
        - **Architecture:** 64-unit LSTM → 32-unit latent vector → RepeatVector → 32/64-unit LSTM decoder → TimeDistributed Dense
        - **Input shape:** `{metadata['seq_len']} × {metadata['n_features']}`
        - **Training:** healthy windows only; unit-level train/validation/test split
        - **Threshold:** `{metadata['threshold_method']}`
        - **Test accuracy:** `79.85%`
        - **Failure recall:** `85.81%`
        - **Failure precision:** `63.98%`
        - **Test ROC-AUC:** `88.67%`
        """
    )

with st.expander("Limitations"):
    st.markdown(
        """
        - The included dataset is synthetic and does not represent a validated physical asset.
        - The global model can be sensitive to differences between equipment units and operating regimes.
        - Reconstruction error identifies unusual patterns; it does not diagnose a root cause or guarantee failure.
        - A production system needs sensor validation, drift monitoring, maintenance labels, operating-condition features, and risk-based threshold governance.
        """
    )

st.subheader("Download scored windows")
st.dataframe(prediction_table, use_container_width=True, height=330)
st.download_button(
    "Download prediction results as CSV",
    data=prediction_table.to_csv(index=False).encode("utf-8"),
    file_name=f"equipment_{selected_unit}_anomaly_predictions.csv",
    mime="text/csv",
)
st.markdown(
    '<p class="small-note">GitHub: replace this placeholder with your repository URL after publishing.</p>',
    unsafe_allow_html=True,
)
