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

from src.config import (
    ATTENTION_QUALIFICATION,
    HEALTHCARE_DISCLAIMER,
    META_PATH,
    METRICS_PATH,
    SAMPLE_DATA_PATH,
    THRESHOLD,
    WEIGHTS_PATH,
)
from src.data_preprocessing import frame_to_sequences, prepare_ecg_frame
from src.inference_pipeline import ECGInferenceService


st.set_page_config(
    page_title="ECG Anomaly Detection",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 ECG Anomaly Detection with LSTM Autoencoder")
st.caption(
    "A portfolio demonstration of normal-only sequence learning, signal reconstruction, "
    "threshold-based anomaly detection, and temporal focus analysis."
)
st.error(HEALTHCARE_DISCLAIMER)


@st.cache_resource
def load_service() -> ECGInferenceService:
    return ECGInferenceService.from_artifacts(
        weights_path=WEIGHTS_PATH,
        threshold=THRESHOLD,
    )


@st.cache_data
def load_packaged_data() -> pd.DataFrame:
    return prepare_ecg_frame(pd.read_csv(SAMPLE_DATA_PATH))


@st.cache_data
def load_metadata() -> tuple[dict, dict]:
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return metadata, metrics


service = load_service()
metadata, metrics = load_metadata()

with st.sidebar:
    st.header("Analysis controls")
    data_source = st.radio(
        "Data source",
        ["Packaged synthetic sample", "Upload CSV"],
    )
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload a wide ECG CSV",
            type=["csv"],
            help="Use one row per signal and 140 numeric signal columns.",
        )

    st.divider()
    st.write(f"**Sequence length:** {metadata['sequence_length']} timesteps")
    st.write(f"**Features:** {metadata['number_of_features']}")
    st.write(f"**Threshold:** {metadata['threshold']:.6f}")
    st.write(f"**Parameters:** {metadata['parameter_count']:,}")
    st.caption("The deployed app loads pretrained NumPy weights and does not retrain.")

try:
    if data_source == "Packaged synthetic sample":
        ecg_frame = load_packaged_data()
    elif uploaded_file is not None:
        ecg_frame = prepare_ecg_frame(pd.read_csv(uploaded_file))
    else:
        st.info("Upload a CSV file or switch to the packaged sample.")
        st.stop()
except Exception as exc:
    st.error(f"Unable to prepare the ECG data: {exc}")
    st.stop()

signal_options = {
    f"{row.signal_id} | label={row.label} | {row.anomaly_type}": index
    for index, row in ecg_frame[["signal_id", "label", "anomaly_type"]].iterrows()
}
selected_label = st.selectbox(
    "Select an ECG signal",
    list(signal_options.keys()),
)
selected_index = signal_options[selected_label]

sequences = frame_to_sequences(ecg_frame)
selected_signal = sequences[selected_index]
selected_result = service.analyze_signal(selected_signal)

status_label = selected_result.predicted_status
status_icon = "⚠️" if selected_result.predicted_label else "✅"

metric_columns = st.columns(4)
metric_columns[0].metric("Prediction", f"{status_icon} {status_label}")
metric_columns[1].metric(
    "Reconstruction error",
    f"{selected_result.reconstruction_error:.6f}",
)
metric_columns[2].metric("Threshold", f"{selected_result.threshold:.6f}")
metric_columns[3].metric("Anomaly score", f"{selected_result.anomaly_score:.2f}×")

if selected_result.predicted_label:
    st.warning(
        "The reconstruction error is above the learned threshold, so this synthetic "
        "sequence is flagged as anomalous. This is not a medical diagnosis."
    )
else:
    st.success(
        "The reconstruction error is below the learned threshold, so this synthetic "
        "sequence is classified as normal by the portfolio model."
    )

signal_tab, dataset_tab, performance_tab, methodology_tab = st.tabs(
    [
        "Signal analysis",
        "Dataset overview",
        "Model performance",
        "Methodology and limitations",
    ]
)

with signal_tab:
    time_axis = np.arange(len(selected_signal))

    comparison = go.Figure()
    comparison.add_trace(
        go.Scatter(
            x=time_axis,
            y=selected_signal[:, 0],
            mode="lines",
            name="Original signal",
        )
    )
    comparison.add_trace(
        go.Scatter(
            x=time_axis,
            y=selected_result.reconstruction,
            mode="lines",
            name="Reconstruction",
        )
    )
    comparison.update_layout(
        title="Original and Reconstructed ECG-Like Signal",
        xaxis_title="Timestep",
        yaxis_title="Signal amplitude",
    )
    st.plotly_chart(comparison, use_container_width=True)

    error_frame = pd.DataFrame(
        {
            "timestep": time_axis,
            "pointwise_error": selected_result.pointwise_error,
            "temporal_focus": selected_result.temporal_focus,
        }
    )

    residual_figure = px.line(
        error_frame,
        x="timestep",
        y="pointwise_error",
        title="Pointwise Absolute Reconstruction Error",
    )
    st.plotly_chart(residual_figure, use_container_width=True)

    focus_figure = px.area(
        error_frame,
        x="timestep",
        y="temporal_focus",
        title="Post-Hoc Temporal Focus",
    )
    st.plotly_chart(focus_figure, use_container_width=True)
    st.info(ATTENTION_QUALIFICATION)

with dataset_tab:
    st.subheader("Prepared dataset")
    st.dataframe(
        ecg_frame[["signal_id", "label", "anomaly_type"]].head(50),
        use_container_width=True,
    )

    with st.spinner("Scoring the selected dataset..."):
        scored_frame = service.score_frame(ecg_frame)

    summary_columns = st.columns(4)
    summary_columns[0].metric("Signals", len(scored_frame))
    summary_columns[1].metric(
        "Flagged anomalies",
        int((scored_frame["predicted_label"] == 1).sum()),
    )
    summary_columns[2].metric(
        "Average error",
        f"{scored_frame['reconstruction_error'].mean():.6f}",
    )
    summary_columns[3].metric(
        "Maximum anomaly score",
        f"{scored_frame['anomaly_score'].max():.2f}×",
    )

    error_distribution = px.histogram(
        scored_frame,
        x="reconstruction_error",
        nbins=45,
        color="predicted_status",
        title="Dataset Reconstruction-Error Distribution",
    )
    error_distribution.add_vline(
        x=THRESHOLD,
        line_dash="dash",
        annotation_text="Threshold",
    )
    st.plotly_chart(error_distribution, use_container_width=True)

    st.dataframe(scored_frame, use_container_width=True)

    st.download_button(
        "Download anomaly predictions",
        data=scored_frame.to_csv(index=False).encode("utf-8"),
        file_name="ecg_anomaly_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

with performance_tab:
    test_metrics = metrics["test"]

    performance_columns = st.columns(6)
    performance_columns[0].metric("Accuracy", f"{test_metrics['accuracy']:.3%}")
    performance_columns[1].metric(
        "Precision",
        f"{test_metrics['precision_anomaly']:.3%}",
    )
    performance_columns[2].metric(
        "Recall",
        f"{test_metrics['recall_anomaly']:.3%}",
    )
    performance_columns[3].metric(
        "F1",
        f"{test_metrics['f1_anomaly']:.3%}",
    )
    performance_columns[4].metric("ROC-AUC", f"{test_metrics['roc_auc']:.4f}")
    performance_columns[5].metric("PR-AUC", f"{test_metrics['pr_auc']:.4f}")

    st.caption(
        "The test set contains 330 synthetic normal signals and 120 synthetic anomaly signals."
    )

    confusion = np.asarray(test_metrics["confusion_matrix"])
    confusion_figure = px.imshow(
        confusion,
        x=["Predicted normal", "Predicted anomaly"],
        y=["True normal", "True anomaly"],
        text_auto=True,
        title="Confusion Matrix",
        aspect="auto",
    )
    st.plotly_chart(confusion_figure, use_container_width=True)

    test_predictions = pd.read_csv(
        PROJECT_ROOT / "outputs" / "test_predictions.csv"
    )
    test_error_figure = px.histogram(
        test_predictions,
        x="reconstruction_error_mae",
        color=test_predictions["true_label"].map(
            {0: "Normal", 1: "Anomaly"}
        ),
        nbins=50,
        title="Held-Out Test Reconstruction Errors",
    )
    test_error_figure.add_vline(
        x=THRESHOLD,
        line_dash="dash",
        annotation_text="Threshold",
    )
    st.plotly_chart(test_error_figure, use_container_width=True)

    baseline = pd.read_csv(
        PROJECT_ROOT / "outputs" / "baseline_comparison.csv"
    )
    baseline_figure = px.bar(
        baseline,
        x="Approach",
        y="F1",
        title="Anomaly-Class F1 Baseline Comparison",
    )
    baseline_figure.update_yaxes(range=[0, 1.05])
    st.plotly_chart(baseline_figure, use_container_width=True)

with methodology_tab:
    st.subheader("Anomaly-detection workflow")
    st.markdown(
        """
1. Generate or load fixed-length ECG-like sequences.
2. Train the autoencoder only on normal training signals.
3. Reconstruct each sequence.
4. Calculate mean absolute reconstruction error across 140 timesteps.
5. Set the threshold to the training-normal mean plus three standard deviations.
6. Flag signals with error at or above the threshold.
7. Evaluate against synthetic labels on untouched validation and test signals.
"""
    )

    st.subheader("Supplied architecture")
    st.code(
        """Input: 140 timesteps × 1 feature
→ LSTM 64, return sequences
→ LSTM 32 latent vector
→ RepeatVector 140
→ LSTM 32, return sequences
→ LSTM 64, return sequences
→ TimeDistributed Dense 1
→ Reconstructed signal""",
        language="text",
    )

    st.subheader("Attention qualification")
    st.info(ATTENTION_QUALIFICATION)

    st.subheader("Known limitations")
    st.markdown(
        """
- The dataset is synthetic and does not represent real clinical ECG recordings.
- The anomaly patterns are deliberately separable, so test performance is optimistic.
- The supplied pretrained model has no trainable attention layer.
- The temporal-focus chart is post-hoc reconstruction-error explainability.
- The threshold may not transfer to other devices, sampling rates, units, or populations.
- The app does not identify arrhythmia types or provide clinical interpretation.
- Real deployment would require governed clinical data, patient-level splitting, calibration,
  external validation, monitoring, and medical-device governance.
"""
    )
    st.error(HEALTHCARE_DISCLAIMER)
