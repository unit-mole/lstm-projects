from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METADATA_PATH, MODEL_PATH, SAMPLE_DATA_PATH
from src.data_preprocessing import load_weather_npz
from src.forecasting_pipeline import recursive_forecast
from src.inference_pipeline import load_metadata, predict_next_frame
from src.model_evaluation import evaluate_weather_map
from src.visualization import comparison_figure, error_heatmap_figure, input_sequence_figure
from src.weather_preprocessing import repair_weather_array

st.set_page_config(page_title="ConvLSTM Weather Forecasting", page_icon="🌦️", layout="wide")


@st.cache_resource(show_spinner="Loading the pretrained ConvLSTM model...")
def get_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def get_sample_data():
    return load_weather_npz(SAMPLE_DATA_PATH)


def load_uploaded_file(uploaded_file):
    raw = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".npz":
        with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
            if "X" not in payload:
                raise ValueError("Uploaded NPZ must contain an X array")
            X = repair_weather_array(payload["X"])
            y = repair_weather_array(payload["y"]) if "y" in payload else None
            future_y = repair_weather_array(payload["future_y"]) if "future_y" in payload else None
    elif suffix == ".npy":
        X = repair_weather_array(np.load(io.BytesIO(raw), allow_pickle=False))
        y = None
        future_y = None
    else:
        raise ValueError("Upload a .npy or .npz file")

    if X.ndim == 4:
        if X.shape[0] == 7:
            y = X[-1][None, ...]
            X = X[:6][None, ...]
        else:
            X = X[None, ...]
    if X.ndim != 5:
        raise ValueError("Expected X shape [samples, 6, 24, 24, 1] or [6, 24, 24, 1]")
    return X, y, future_y


def npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def forecast_gif_bytes(frames: np.ndarray) -> bytes:
    images = []
    for index, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(frame[:, :, 0], vmin=0, vmax=1)
        ax.set_title(f"Forecast +{index + 1} step")
        ax.axis("off")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        images.append(imageio.imread(buffer))
    output = io.BytesIO()
    imageio.mimsave(output, images, format="GIF", duration=0.8, loop=0)
    return output.getvalue()


metadata = load_metadata(METADATA_PATH)

st.title("Weather Forecasting using ConvLSTM")
st.caption("Spatiotemporal next-frame prediction from six historical 24 × 24 weather-intensity grids")
st.warning(
    "Responsible use: This educational portfolio project is not an official weather forecasting system. "
    "Do not use its forecasts for emergency, aviation, agriculture, transportation, safety-critical, or operational decisions."
)

with st.sidebar:
    st.header("Forecast Controls")
    source = st.radio("Data source", ["Portfolio sample", "Upload .npy / .npz"])
    horizon = st.slider("Recursive forecast horizon", 1, 6, 4)
    threshold = st.slider("Weather-event threshold", 0.10, 0.90, float(metadata["metrics"]["threshold"]), 0.05)
    st.markdown("**Required input shape:** `6 × 24 × 24 × 1`")
    st.markdown("[GitHub repository](https://github.com/unit-mole/lstm-projects)")

try:
    if source == "Portfolio sample":
        X, y, future_y = get_sample_data()
    else:
        upload = st.file_uploader("Upload weather sequence", type=["npy", "npz"])
        if upload is None:
            st.info("Upload a NumPy weather sequence file to continue.")
            st.stop()
        X, y, future_y = load_uploaded_file(upload)

    expected_tail = (metadata["input_frames"], metadata["height"], metadata["width"], metadata["channels"])
    if tuple(X.shape[1:]) != expected_tail:
        raise ValueError(f"Model expects [samples, {expected_tail}], received {X.shape}")

    sample_index = st.sidebar.number_input("Sequence index", 0, len(X) - 1, 0, 1)
    sequence = X[int(sample_index)]
    actual = y[int(sample_index)] if y is not None and int(sample_index) < len(y) else None

    overview_1, overview_2, overview_3, overview_4 = st.columns(4)
    overview_1.metric("Input frames", metadata["input_frames"])
    overview_2.metric("Grid size", f'{metadata["height"]} × {metadata["width"]}')
    overview_3.metric("Channels", metadata["channels"])
    overview_4.metric("Model parameters", f'{metadata["model_parameters"]:,}')

    st.subheader("Input Weather Sequence")
    st.pyplot(input_sequence_figure(sequence), use_container_width=True)

    with st.expander("Sequence data summary"):
        st.dataframe(pd.DataFrame({
            "frame": np.arange(1, len(sequence) + 1),
            "mean_intensity": sequence.mean(axis=(1, 2, 3)),
            "maximum_intensity": sequence.max(axis=(1, 2, 3)),
            "active_pixel_share": (sequence >= threshold).mean(axis=(1, 2, 3)),
        }), use_container_width=True)

    if st.button("Generate Weather Forecast", type="primary", use_container_width=True):
        model = get_model()
        with st.spinner("Running ConvLSTM inference..."):
            predicted = predict_next_frame(model, sequence, metadata)
            forecasts = recursive_forecast(model, sequence, metadata, steps=horizon)

        st.subheader("Next-Frame Forecast")
        st.pyplot(comparison_figure(sequence[-1], actual, predicted), use_container_width=True)

        if actual is not None:
            result_metrics = evaluate_weather_map(actual, predicted, threshold)
            cols = st.columns(6)
            for column, (name, value) in zip(cols, [
                ("MAE", result_metrics["mae"]),
                ("RMSE", result_metrics["rmse"]),
                ("SSIM", result_metrics["ssim"]),
                ("IoU / CSI", result_metrics["iou"]),
                ("POD", result_metrics["pod"]),
                ("FAR", result_metrics["far"]),
            ]):
                column.metric(name, f"{value:.4f}")
            st.pyplot(error_heatmap_figure(actual, predicted), use_container_width=True)

        st.subheader("Recursive Multi-Step Forecast")
        forecast_cols = st.columns(min(horizon, 3))
        for index, frame in enumerate(forecasts):
            column = forecast_cols[index % len(forecast_cols)]
            fig, ax = plt.subplots(figsize=(4, 3.5))
            ax.imshow(frame[:, :, 0], vmin=0, vmax=1)
            ax.set_title(f"+{index + 1} step")
            ax.axis("off")
            fig.tight_layout()
            column.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if future_y is not None and int(sample_index) < len(future_y):
            available = min(horizon, future_y.shape[1])
            multi_mae = float(np.mean(np.abs(future_y[int(sample_index), :available] - forecasts[:available])))
            st.info(f"Mean absolute error across the first {available} recursive step(s): {multi_mae:.4f}")

        csv_frame = pd.DataFrame({
            "grid_row": np.repeat(np.arange(metadata["height"]), metadata["width"]),
            "grid_col": np.tile(np.arange(metadata["width"]), metadata["height"]),
            "predicted_intensity": predicted[:, :, 0].reshape(-1),
        })
        download_1, download_2, download_3 = st.columns(3)
        download_1.download_button("Download next frame (.npy)", npy_bytes(predicted), "predicted_weather_frame.npy")
        download_2.download_button("Download next frame (.csv)", csv_frame.to_csv(index=False), "predicted_weather_frame.csv", "text/csv")
        download_3.download_button("Download forecast GIF", forecast_gif_bytes(forecasts), "weather_forecast_sequence.gif", "image/gif")

    st.divider()
    st.subheader("Recorded Held-Out Test Results")
    metric_rows = [
        ["Persistence baseline", metadata["metrics"]["baseline_test_mae"], metadata["metrics"]["baseline_test_rmse"]],
        ["ConvLSTM", metadata["metrics"]["convlstm_test_mae"], metadata["metrics"]["convlstm_test_rmse"]],
    ]
    st.dataframe(pd.DataFrame(metric_rows, columns=["Model", "Test MAE", "Test RMSE"]), use_container_width=True, hide_index=True)
    st.success(
        f"The supplied ConvLSTM artifact reduced test MAE by {metadata['metrics']['mae_improvement_percent']:.1f}% "
        f"and test RMSE by {metadata['metrics']['rmse_improvement_percent']:.1f}% relative to persistence."
    )

    with st.expander("How ConvLSTM works"):
        st.write(
            "A normal LSTM learns temporal patterns from vectors, while a CNN learns spatial patterns from grids or images. "
            "ConvLSTM places convolution operations inside the recurrent memory mechanism so it can learn movement, shape, "
            "intensity, and temporal evolution together."
        )
    with st.expander("Limitations"):
        st.markdown(
            "- The bundled data is synthetic and low resolution.\n"
            "- Recursive forecasts accumulate error because predicted frames become future inputs.\n"
            "- Sudden regime changes can be smoothed or missed.\n"
            "- This project does not replace numerical weather prediction, official meteorological data, or expert review."
        )
except Exception as error:
    st.error(f"Unable to run the forecasting workflow: {error}")
    st.exception(error)
