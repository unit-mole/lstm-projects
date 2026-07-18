from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LOOKBACK, METADATA_PATH, MODEL_PATH, SAMPLE_DATA_PATH, SCALER_PATH, SEASONAL_PERIOD
from src.data_preprocessing import load_and_prepare
from src.forecasting_pipeline import (
    evaluate_history,
    load_artifacts,
    recursive_forecast,
    summarize_forecast,
)


st.set_page_config(
    page_title="Airline Passenger Forecasting",
    page_icon="✈️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the trained LSTM model...")
def cached_artifacts():
    return load_artifacts(MODEL_PATH, SCALER_PATH, METADATA_PATH)


@st.cache_data(show_spinner=False)
def cached_sample_data():
    return load_and_prepare(SAMPLE_DATA_PATH)


def trend_figure(frame: pd.DataFrame, forecast: pd.DataFrame | None = None):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["Month"], y=frame["Passengers"], mode="lines", name="Historical"
        )
    )
    if forecast is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast["Month"],
                y=forecast["Forecasted_Passengers"],
                mode="lines+markers",
                name="Forecast",
                line={"dash": "dash"},
            )
        )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Passengers (thousands)",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
    )
    return fig


def main() -> None:
    st.title("✈️ Airline Passenger Forecasting with LSTM")
    st.caption(
        "A deployment-ready demand forecasting demo using a seasonally adjusted "
        "LSTM and chronological time-series evaluation."
    )

    model, scaler, metadata = cached_artifacts()

    with st.sidebar:
        st.header("Forecast controls")
        source = st.radio("Data source", ["Use sample dataset", "Upload CSV"])
        uploaded = None
        if source == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload monthly passenger history",
                type=["csv"],
                help="Expected columns resemble Month and Passengers. At least 24 months are required.",
            )
        horizon = st.select_slider(
            "Forecast horizon",
            options=[6, 12, 18, 24],
            value=12,
        )
        st.divider()
        st.write(f"**LSTM input window:** {LOOKBACK} seasonal-growth observations")
        st.write(f"**Effective raw history:** {LOOKBACK + SEASONAL_PERIOD} months")
        st.write("**Forecast frequency:** Monthly")

    try:
        if source == "Use sample dataset" or uploaded is None:
            frame, notes = cached_sample_data()
            dataset_label = "AirPassengers sample"
        else:
            frame, notes = load_and_prepare(uploaded)
            dataset_label = uploaded.name

        if len(frame) < LOOKBACK + SEASONAL_PERIOD:
            st.error(
                f"The model requires at least {LOOKBACK + SEASONAL_PERIOD} consecutive monthly observations."
            )
            st.stop()

        forecast = recursive_forecast(
            frame,
            model,
            scaler,
            horizon=horizon,
            lookback=LOOKBACK,
            seasonal_period=SEASONAL_PERIOD,
        )
        summary = summarize_forecast(forecast)

        tab_overview, tab_data, tab_performance, tab_forecast = st.tabs(
            ["Overview", "Data & seasonality", "Model performance", "Future forecast"]
        )

        with tab_overview:
            st.subheader(dataset_label)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("History", f"{len(frame)} months")
            c2.metric("Forecast horizon", f"{horizon} months")
            c3.metric("Average forecast", f"{summary['average_forecast']:,.0f}")
            c4.metric("Trend", str(summary["trend_direction"]).title())
            st.plotly_chart(trend_figure(frame, forecast), width="stretch")
            st.info(summary["business_interpretation"])
            if notes:
                with st.expander("Preprocessing notes"):
                    for note in notes:
                        st.write(f"- {note}")

        with tab_data:
            st.dataframe(frame[["Month", "Passengers"]].tail(24), width="stretch")
            seasonal = (
                frame.groupby(["MonthNumber", "MonthName"], as_index=False)["Passengers"]
                .mean()
                .sort_values("MonthNumber")
            )
            seasonal_fig = px.line(
                seasonal,
                x="MonthName",
                y="Passengers",
                markers=True,
                title="Average passenger demand by calendar month",
            )
            seasonal_fig.update_xaxes(categoryorder="array", categoryarray=seasonal["MonthName"].tolist())
            st.plotly_chart(seasonal_fig, width="stretch")
            st.caption(
                "The model learns year-over-year log growth while the 12-month anchor "
                "preserves the repeating seasonal demand pattern."
            )

        with tab_performance:
            if source == "Use sample dataset":
                metrics = metadata["test_metrics"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Test MAE", f"{metrics['mae']:.2f}")
                c2.metric("Test RMSE", f"{metrics['rmse']:.2f}")
                c3.metric("Test MAPE", f"{metrics['mape']:.2f}%")
                c4.metric("Test R²", f"{metrics['r2']:.3f}")
                predictions = pd.read_csv(PROJECT_ROOT / "outputs" / "test_predictions.csv", parse_dates=["Month"])
            else:
                if len(frame) <= LOOKBACK + SEASONAL_PERIOD:
                    predictions = None
                    st.warning(
                        "At least 25 months are needed to calculate historical one-step "
                        "performance. Future forecasting is still available with 24 months."
                    )
                else:
                    predictions, metrics_obj = evaluate_history(
                        frame, model, scaler, LOOKBACK, SEASONAL_PERIOD
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Historical MAE", f"{metrics_obj.mae:.2f}")
                    c2.metric("Historical RMSE", f"{metrics_obj.rmse:.2f}")
                    c3.metric("Historical MAPE", f"{metrics_obj.mape:.2f}%")
                    c4.metric("Historical R²", f"{metrics_obj.r2:.3f}")

            if predictions is not None:
                perf_fig = go.Figure()
                perf_fig.add_trace(go.Scatter(x=predictions["Month"], y=predictions["Actual"], name="Actual"))
                perf_fig.add_trace(go.Scatter(x=predictions["Month"], y=predictions["Predicted"], name="Predicted", line={"dash": "dash"}))
                perf_fig.update_layout(
                    title="One-step-ahead actual vs predicted",
                    xaxis_title="Month",
                    yaxis_title="Passengers (thousands)",
                    hovermode="x unified",
                )
                st.plotly_chart(perf_fig, width="stretch")
                st.caption(
                    "For the packaged sample model, metrics are calculated only on the final "
                    "24 chronologically held-out months. Uploaded-series metrics are descriptive "
                    "and do not represent a retrained model."
                )

        with tab_forecast:
            st.plotly_chart(trend_figure(frame.tail(60), forecast), width="stretch")
            display_forecast = forecast.copy()
            display_forecast["Forecasted_Passengers"] = display_forecast["Forecasted_Passengers"].round(1)
            st.dataframe(display_forecast, width="stretch", hide_index=True)
            csv_bytes = display_forecast.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download forecast CSV",
                data=csv_bytes,
                file_name=f"airline_passenger_forecast_{horizon}_months.csv",
                mime="text/csv",
            )
            st.markdown(
                "**Business use:** Capacity planning, aircraft scheduling, staffing, "
                "seasonal campaign planning, and revenue target setting."
            )

        st.divider()
        st.caption(
            "Portfolio project only. Forecasts are based on historical passenger counts and "
            "do not include fares, macroeconomic variables, route capacity, fuel prices, or disruptions."
        )

    except Exception as exc:
        st.error(f"Unable to generate the forecast: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
