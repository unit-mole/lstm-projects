"""Interactive portfolio demo for ConvLSTM next-frame prediction."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METADATA_PATH, MODEL_PATH, SAMPLE_DATA_PATH
from src.data_preprocessing import load_sample_sequences
from src.inference_pipeline import run_inference
from src.model_evaluation import calculate_frame_metrics
from src.prediction_pipeline import load_prediction_model
from src.video_preprocessing import preprocess_frame_zip_bytes, preprocess_video_bytes
from src.visualization import (
    frame_to_png_bytes,
    frames_to_gif_bytes,
    plot_input_sequence,
    plot_prediction_comparison,
)

st.set_page_config(
    page_title="ConvLSTM Video Frame Prediction",
    page_icon="🎞️",
    layout="wide",
)

RESPONSIBLE_USE = (
    "This project is for educational and portfolio demonstration purposes only. "
    "Predicted frames are model-generated estimates and may be blurry, inaccurate, or unrealistic. "
    "Do not use this demo for surveillance, safety-critical monitoring, medical imaging, autonomous "
    "driving, legal decisions, or production video analytics without rigorous validation. Do not upload "
    "private, sensitive, copyrighted, or personally identifiable video content."
)


@st.cache_resource(show_spinner="Loading the ConvLSTM model...")
def get_model():
    return load_prediction_model(MODEL_PATH)


@st.cache_data
def get_samples() -> dict[str, np.ndarray]:
    return load_sample_sequences(SAMPLE_DATA_PATH)


@st.cache_data
def get_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


st.title("🎞️ Video Frame Prediction using Convolutional LSTM")
st.caption(
    "Predict the next grayscale frame from six ordered observations and explore recursive future-frame generation."
)
st.warning(RESPONSIBLE_USE)

metadata = get_metadata()
model = get_model()

with st.sidebar:
    st.header("Prediction controls")
    source_mode = st.radio(
        "Input source",
        ["Preloaded safe sample", "Upload short video", "Upload ZIP of frames"],
    )
    future_steps = st.slider("Recursive future frames", 1, 6, 2)
    st.caption("The trained model has a fixed input length of 6 frames at 32 × 32 grayscale resolution.")

sequence: np.ndarray | None = None
actual: np.ndarray | None = None
source_note = ""

if source_mode == "Preloaded safe sample":
    samples = get_samples()
    sample_index = st.sidebar.selectbox(
        "Sample sequence",
        options=list(range(len(samples["X"]))),
        format_func=lambda value: f"Sequence {value + 1}",
    )
    sequence = samples["X"][sample_index]
    actual = samples["y"][sample_index]
    source_note = "Synthetic moving-square sequence bundled for safe public demonstration."
elif source_mode == "Upload short video":
    upload = st.file_uploader("Upload a short MP4, MOV, AVI, or MKV video", type=["mp4", "mov", "avi", "mkv"])
    stride = st.slider("Frame stride", 1, 5, 1, help="Use every nth frame while preserving order.")
    if upload is not None:
        try:
            sequence, actual, upload_meta = preprocess_video_bytes(
                upload.getvalue(), upload.name, input_frames=6, target_size=(32, 32), stride=stride
            )
            source_note = (
                f"Read {upload_meta['frames_read']} ordered frames; reported video FPS: "
                f"{upload_meta['reported_fps']:.2f}; stride: {upload_meta['stride']}."
            )
        except Exception as exc:
            st.error(f"The uploaded video could not be prepared: {exc}")
elif source_mode == "Upload ZIP of frames":
    upload = st.file_uploader("Upload a ZIP containing ordered image files", type=["zip"])
    if upload is not None:
        try:
            sequence, actual, filenames = preprocess_frame_zip_bytes(
                upload.getvalue(), input_frames=6, target_size=(32, 32)
            )
            source_note = "Loaded frames in filename order: " + ", ".join(filenames)
        except Exception as exc:
            st.error(f"The ZIP archive could not be prepared: {exc}")

if sequence is None:
    st.info("Choose the safe sample or upload data to enable prediction.")
    st.stop()

st.write(source_note)

input_tab, prediction_tab, model_tab, limitations_tab = st.tabs(
    ["Input sequence", "Prediction results", "Model details", "Limitations"]
)

with input_tab:
    st.subheader("Ordered input frames")
    input_figure = plot_input_sequence(sequence)
    st.pyplot(input_figure, use_container_width=True)
    plt.close(input_figure)
    st.code(f"Model input shape: {sequence[None, ...].shape}")

with prediction_tab:
    if st.button("Generate prediction", type="primary", use_container_width=True):
        with st.spinner("Running ConvLSTM inference..."):
            result = run_inference(
                model,
                sequence,
                actual_next_frame=actual,
                future_steps=future_steps,
            )
        comparison = plot_prediction_comparison(
            result.input_sequence,
            result.predicted_next_frame,
            result.actual_next_frame,
        )
        st.pyplot(comparison, use_container_width=True)
        plt.close(comparison)

        if result.metrics:
            columns = st.columns(4)
            columns[0].metric("MAE", f"{result.metrics['mae']:.4f}")
            columns[1].metric("RMSE", f"{result.metrics['rmse']:.4f}")
            columns[2].metric("SSIM", f"{result.metrics['ssim']:.4f}")
            columns[3].metric("IoU", f"{result.metrics['iou']:.4f}")
            with st.expander("All single-sequence metrics"):
                st.dataframe(pd.DataFrame([result.metrics]).T.rename(columns={0: "Value"}))
        else:
            st.info("No actual next frame was available, so error metrics cannot be calculated.")

        st.subheader("Recursive future-frame sequence")
        st.image(
            [frame[..., 0] for frame in result.predicted_future_frames],
            caption=[f"Forecast +{i + 1}" for i in range(len(result.predicted_future_frames))],
            clamp=True,
        )
        st.caption(
            "For horizons beyond one frame, each prediction is fed back into the rolling input window. "
            "This is convenient but can compound errors and blur motion over time."
        )
        download_one, download_gif = st.columns(2)
        download_one.download_button(
            "Download predicted next frame (PNG)",
            data=frame_to_png_bytes(result.predicted_next_frame),
            file_name="predicted_next_frame.png",
            mime="image/png",
            use_container_width=True,
        )
        download_gif.download_button(
            "Download recursive forecast (GIF)",
            data=frames_to_gif_bytes(result.predicted_future_frames),
            file_name="predicted_future_sequence.gif",
            mime="image/gif",
            use_container_width=True,
        )
    else:
        st.info("Select **Generate prediction** to run the saved model. The app never retrains on startup.")

with model_tab:
    st.subheader("What ConvLSTM learns")
    st.markdown(
        "A standard LSTM learns temporal patterns from vector sequences, while a CNN learns spatial "
        "patterns from images. ConvLSTM replaces the dense operations inside an LSTM with convolutions, "
        "allowing the model to preserve image structure while learning motion through time."
    )
    st.json(metadata["model"])
    st.subheader("Test-set comparison")
    metric_rows = []
    for name, values in metadata["metrics"].items():
        metric_rows.append({"Approach": name, **values})
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)
    st.caption(
        "SSIM can favor the persistence baseline on this sparse black-background dataset even when "
        "ConvLSTM has much lower MAE/RMSE and substantially higher foreground IoU. Metrics should be interpreted together."
    )

with limitations_tab:
    st.markdown(
        """
        - The model was trained on synthetic 32 × 32 grayscale moving-square sequences, not complex real-world scenes.
        - Sudden motion changes, occlusion, camera movement, multiple objects, and texture are outside the training distribution.
        - Recursive predictions accumulate error because generated frames become future inputs.
        - Pixel accuracy is inflated by the large black background; foreground IoU and visual inspection are more informative.
        - Modern video-prediction systems may use transformers, diffusion models, or larger recurrent architectures.
        - This is an educational deployment, not a production video-generation or monitoring system.
        """
    )

st.divider()
st.caption("Portfolio project • Keras 3 ConvLSTM • JAX inference backend • Streamlit")
