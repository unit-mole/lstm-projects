"""Portfolio Streamlit application for Stacked LSTM traffic forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SAMPLE_DATA  # noqa: E402
from src.config import MODEL_DIR  # noqa: E402
from src.data_preprocessing import load_traffic_csv  # noqa: E402
from src.data_preprocessing import missing_value_summary  # noqa: E402
from src.data_preprocessing import prepare_traffic_data  # noqa: E402
from src.forecasting_pipeline import TrafficForecastingPipeline  # noqa: E402


st.set_page_config(
    page_title="Traffic Flow Prediction | Stacked LSTM",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Flow Prediction using Stacked LSTM")
st.caption(
    "An artifact-backed smart-mobility forecasting demo for hourly "
    "traffic congestion patterns."
)

st.warning(
    "Responsible use: This educational portfolio project must not be used "
    "as the sole basis for traffic control, public safety, emergency "
    "response, transportation policy, or operational decisions. "
    "Forecasts require validated data, domain expertise, infrastructure "
    "context, and continuous monitoring."
)


@st.cache_resource
def load_pipeline() -> TrafficForecastingPipeline:
    return TrafficForecastingPipeline.from_artifacts(MODEL_DIR)


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_SAMPLE_DATA)


pipeline = load_pipeline()

with st.sidebar:
    st.header("Forecast controls")
    data_source = st.radio(
        "Data source",
        ["Preloaded sample", "Upload CSV"],
    )
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload hourly traffic data",
            type=["csv"],
        )

    horizon = st.slider(
        "Scenario forecast horizon",
        min_value=1,
        max_value=24,
        value=12,
        help=(
            "The saved model is one-step ahead. Multi-step output is "
            "generated recursively with seasonal external-input scenarios."
        ),
    )
    st.markdown(
        "[View source repository]"
        "(https://github.com/unit-mole/lstm-projects/tree/main/"
        "10-traffic-flow-prediction-stacked-lstm)"
    )

try:
    if data_source == "Preloaded sample":
        raw_data = load_sample()
    elif uploaded_file is not None:
        raw_data = load_traffic_csv(uploaded_file)
    else:
        st.info("Upload a CSV file or switch to the preloaded sample.")
        st.stop()

    prepared = prepare_traffic_data(raw_data)
except (ValueError, OSError, pd.errors.ParserError) as error:
    st.error(str(error))
    st.stop()

overview_tab, patterns_tab, backtest_tab, forecast_tab, model_tab = st.tabs(
    [
        "Overview",
        "Traffic patterns",
        "Backtest",
        "Future forecast",
        "Model details",
    ]
)

with overview_tab:
    first_timestamp = prepared["timestamp"].min()
    last_timestamp = prepared["timestamp"].max()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(prepared):,}")
    metric_columns[1].metric(
        "Date range",
        f"{first_timestamp:%Y-%m-%d} → {last_timestamp:%Y-%m-%d}",
    )
    metric_columns[2].metric(
        "Average congestion",
        f"{prepared['congestion_index'].mean():.1f}",
    )
    metric_columns[3].metric(
        "Peak congestion",
        f"{prepared['congestion_index'].max():.1f}",
    )

    st.subheader("Data preview")
    st.dataframe(prepared.head(20), use_container_width=True)

    with st.expander("Missing-value summary"):
        st.dataframe(
            missing_value_summary(raw_data),
            use_container_width=True,
        )

    st.markdown(
        "**Required model columns:** `timestamp`, `vehicle_count`, "
        "`avg_speed`, `occupancy`, `weather_severity`, and "
        "`congestion_index`."
    )

with patterns_tab:
    st.subheader("Multivariate traffic trend")
    selected_signal = st.selectbox(
        "Signal",
        [
            "congestion_index",
            "vehicle_count",
            "avg_speed",
            "occupancy",
            "weather_severity",
        ],
    )
    trend_figure = px.line(
        prepared,
        x="timestamp",
        y=selected_signal,
        title=f"{selected_signal.replace('_', ' ').title()} over time",
    )
    st.plotly_chart(trend_figure, use_container_width=True)

    hourly = (
        prepared.groupby("hour", as_index=False)[
            ["congestion_index", "vehicle_count", "avg_speed"]
        ]
        .mean()
    )
    hourly_figure = px.line(
        hourly,
        x="hour",
        y="congestion_index",
        markers=True,
        title="Average congestion by hour of day",
    )
    st.plotly_chart(hourly_figure, use_container_width=True)

    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    daily = (
        prepared.groupby("dayofweek", as_index=False)[
            "congestion_index"
        ]
        .mean()
    )
    daily["day"] = daily["dayofweek"].map(day_names)
    daily_figure = px.bar(
        daily,
        x="day",
        y="congestion_index",
        title="Average congestion by day of week",
    )
    st.plotly_chart(daily_figure, use_container_width=True)

with backtest_tab:
    if len(prepared) <= pipeline.scaling.sequence_length:
        st.info(
            f"At least {pipeline.scaling.sequence_length + 1} rows are "
            "required for backtesting."
        )
    else:
        with st.spinner("Running artifact-backed one-step backtest..."):
            backtest = pipeline.backtest(prepared)

        metrics = backtest["model_metrics"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("MAE", f"{metrics['mae']:.3f}")
        metric_columns[1].metric("RMSE", f"{metrics['rmse']:.3f}")
        metric_columns[2].metric(
            "MAPE",
            f"{metrics['mape_pct']:.2f}%",
        )
        metric_columns[3].metric("R²", f"{metrics['r2']:.3f}")

        predictions = backtest["predictions"]
        visible_predictions = predictions.tail(min(len(predictions), 500))
        actual_predicted_figure = go.Figure()
        actual_predicted_figure.add_trace(
            go.Scatter(
                x=visible_predictions["timestamp"],
                y=visible_predictions["actual_congestion_index"],
                mode="lines",
                name="Actual",
            )
        )
        actual_predicted_figure.add_trace(
            go.Scatter(
                x=visible_predictions["timestamp"],
                y=visible_predictions["predicted_congestion_index"],
                mode="lines",
                name="Stacked LSTM",
            )
        )
        actual_predicted_figure.update_layout(
            title="Actual vs predicted congestion",
            xaxis_title="Timestamp",
            yaxis_title="Congestion index",
        )
        st.plotly_chart(
            actual_predicted_figure,
            use_container_width=True,
        )

        residual_figure = px.scatter(
            visible_predictions,
            x="timestamp",
            y="residual",
            title="Forecast residuals",
        )
        residual_figure.add_hline(y=0)
        st.plotly_chart(residual_figure, use_container_width=True)

        st.subheader("Baseline comparison")
        comparison = backtest["comparison"].copy()
        st.dataframe(
            comparison.style.format(
                {
                    "mae": "{:.3f}",
                    "rmse": "{:.3f}",
                    "mape_pct": "{:.2f}",
                    "r2": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

        st.download_button(
            "Download backtest predictions",
            predictions.to_csv(index=False).encode("utf-8"),
            file_name="traffic_backtest_predictions.csv",
            mime="text/csv",
        )

with forecast_tab:
    if len(prepared) < pipeline.scaling.sequence_length:
        st.info(
            f"At least {pipeline.scaling.sequence_length} rows are "
            "required for forecasting."
        )
    else:
        with st.spinner("Generating recursive scenario forecast..."):
            future = pipeline.recursive_forecast(
                prepared,
                horizon=horizon,
            )

        first_prediction = future["predicted_congestion_index"].iloc[0]
        highest_prediction = future["predicted_congestion_index"].max()
        peak_row = future.loc[
            future["predicted_congestion_index"].idxmax()
        ]
        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "Next-step congestion",
            f"{first_prediction:.1f}",
            pipeline.congestion_label(first_prediction),
        )
        metric_columns[1].metric(
            "Highest forecast",
            f"{highest_prediction:.1f}",
        )
        metric_columns[2].metric(
            "Expected peak",
            pd.Timestamp(peak_row["timestamp"]).strftime("%a %H:%M"),
            peak_row["traffic_band"],
        )

        recent = prepared.tail(24 * 7)[
            ["timestamp", "congestion_index"]
        ].rename(columns={"congestion_index": "value"})
        recent["series"] = "Historical"
        forecast_plot_data = future[
            ["timestamp", "predicted_congestion_index"]
        ].rename(columns={"predicted_congestion_index": "value"})
        forecast_plot_data["series"] = "Forecast"
        combined = pd.concat(
            [recent, forecast_plot_data],
            ignore_index=True,
        )
        forecast_figure = px.line(
            combined,
            x="timestamp",
            y="value",
            color="series",
            title="Recent history and scenario-based traffic forecast",
        )
        st.plotly_chart(forecast_figure, use_container_width=True)
        st.dataframe(future, use_container_width=True)

        st.info(
            "The packaged Stacked LSTM is a one-step model. Forecasts "
            "beyond the first step are recursive. Vehicle count, speed, "
            "occupancy, and weather use recent hour-of-week seasonal "
            "profiles, so the multi-step view is a planning scenario—not "
            "a validated operational forecast."
        )

        st.download_button(
            "Download future forecast",
            future.to_csv(index=False).encode("utf-8"),
            file_name="traffic_future_forecast.csv",
            mime="text/csv",
        )

with model_tab:
    metadata = pipeline.metadata
    st.subheader("Packaged model")
    st.code(
        "24 hourly observations × 10 features\n"
        "→ LSTM(64, return_sequences=True)\n"
        "→ Dropout(0.20)\n"
        "→ LSTM(32, return_sequences=True)\n"
        "→ Dropout(0.20)\n"
        "→ LSTM(16)\n"
        "→ Dense(16, ReLU)\n"
        "→ Dense(1, linear congestion forecast)"
    )

    packaged_metrics = metadata.get("test_metrics", {})
    if packaged_metrics:
        st.subheader("Original held-out test results")
        st.json(packaged_metrics)

    st.subheader("Input features")
    st.write(", ".join(pipeline.scaling.feature_columns))

    st.subheader("Limitations")
    st.markdown(
        "- The training dataset is deterministic and synthetic.\n"
        "- The model predicts one step ahead directly.\n"
        "- Recursive forecasts accumulate uncertainty.\n"
        "- Road incidents, events, construction, and real weather feeds "
        "are not modeled.\n"
        "- Results require real-world validation before operational use."
    )
