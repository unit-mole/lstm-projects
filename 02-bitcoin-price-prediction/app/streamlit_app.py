from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud_inference import NumpyBitcoinLSTM
from src.config import (
    CONFIG_PATH,
    FINANCIAL_DISCLAIMER,
    METADATA_PATH,
    SAMPLE_DATA_PATH,
    SCALER_PATH,
    SUPPORTED_HORIZONS,
    WEIGHTS_PATH,
)
from src.data_preprocessing import (
    clean_market_data,
    fetch_optional_yfinance,
    load_csv,
    validate_history_length,
)
from src.feature_engineering import create_market_features
from src.forecasting_pipeline import (
    forecast_future,
    load_json,
    load_scaler,
    replay_predictions,
)
from src.model_evaluation import regression_metrics


st.set_page_config(
    page_title="Bitcoin Price Prediction using LSTM",
    page_icon="₿",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
      .financial-warning {
          border: 1px solid #f59e0b;
          border-left: 6px solid #f59e0b;
          border-radius: 0.5rem;
          padding: 0.9rem 1rem;
          background: rgba(245, 158, 11, 0.08);
          margin-bottom: 1rem;
      }
      .small-note {font-size: 0.90rem; opacity: 0.82;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = NumpyBitcoinLSTM(WEIGHTS_PATH)
    scaler = load_scaler(SCALER_PATH)
    config = load_json(CONFIG_PATH)
    metadata = load_json(METADATA_PATH)
    return model, scaler, config, metadata


@st.cache_data
def load_sample() -> pd.DataFrame:
    return clean_market_data(pd.read_csv(SAMPLE_DATA_PATH))


def currency(value: float) -> str:
    return f"${value:,.2f}"


def historical_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="Bitcoin Close",
            mode="lines",
        )
    )
    fig.update_layout(
        title="Historical Bitcoin Closing Price",
        xaxis_title="Date",
        yaxis_title="Closing price (USD)",
        hovermode="x unified",
        height=480,
    )
    return fig


def moving_average_figure(features: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column in ["Close", "SMA_7", "SMA_30"]:
        fig.add_trace(
            go.Scatter(
                x=features["Date"],
                y=features[column],
                mode="lines",
                name=column,
            )
        )
    fig.update_layout(
        title="Closing Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="USD",
        hovermode="x unified",
        height=470,
    )
    return fig


def forecast_figure(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    history_tail = history.tail(180)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_tail["Date"],
            y=history_tail["Close"],
            mode="lines",
            name="Historical Close",
        )
    )
    bridge_dates = [history_tail["Date"].iloc[-1], *forecast["Date"].tolist()]
    bridge_values = [history_tail["Close"].iloc[-1], *forecast["Predicted_Close"].tolist()]
    fig.add_trace(
        go.Scatter(
            x=bridge_dates,
            y=bridge_values,
            mode="lines+markers",
            name="LSTM Forecast",
        )
    )
    fig.add_vline(x=history_tail["Date"].iloc[-1], line_dash="dash")
    fig.update_layout(
        title="Historical Close and Recursive LSTM Forecast",
        xaxis_title="Date",
        yaxis_title="Closing price (USD)",
        hovermode="x unified",
        height=500,
    )
    return fig


def prediction_figure(predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Actual_Close"],
            name="Actual",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predictions["Date"],
            y=predictions["Predicted_Close"],
            name="Predicted",
            mode="lines",
        )
    )
    fig.update_layout(
        title="One-Step Replay: Actual vs Predicted Close",
        xaxis_title="Date",
        yaxis_title="USD",
        hovermode="x unified",
        height=480,
    )
    return fig


model, scaler, config, metadata = load_artifacts()
look_back = int(config.get("look_back", 30))

st.title("₿ Bitcoin Price Prediction using LSTM")
st.caption(
    "A portfolio demonstration of multivariate daily Bitcoin price forecasting, "
    "model evaluation, recursive multi-step inference, and deployment."
)

st.markdown(
    f'<div class="financial-warning"><strong>Financial disclaimer:</strong> '
    f'{FINANCIAL_DISCLAIMER} Consult a qualified financial professional before '
    f'making investment decisions.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Forecast Controls")
    data_source = st.radio(
        "Data source",
        [
            "Packaged offline sample",
            "Upload CSV",
            "Optional recent BTC-USD data",
        ],
        help="The packaged sample keeps the app functional without external APIs.",
    )
    horizon = st.select_slider(
        "Forecast horizon",
        options=SUPPORTED_HORIZONS,
        value=7,
        format_func=lambda value: f"{value} day" if value == 1 else f"{value} days",
    )
    st.metric("LSTM input window", f"{look_back} days")
    st.caption("Target: next daily closing price")
    st.divider()
    st.caption("Model features")
    st.code("Close\nSMA_7\nSMA_30\nDaily Return\nVolume", language=None)

try:
    source_note = ""
    if data_source == "Packaged offline sample":
        market = load_sample()
        source_note = (
            "Deterministic offline demonstration dataset with the same OHLCV schema "
            "as the original Yahoo Finance workflow."
        )
    elif data_source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Upload Bitcoin market CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to continue, or choose the packaged sample.")
            st.stop()
        market = clean_market_data(load_csv(uploaded))
        source_note = "User-uploaded CSV; no data is retained by the application."
    else:
        try:
            market = fetch_optional_yfinance(ticker="BTC-USD", period="2y")
            source_note = "Recent BTC-USD data retrieved through yfinance."
        except Exception as exc:
            st.warning(
                "Recent data could not be retrieved, so the packaged offline sample "
                f"was loaded instead. Details: {exc}"
            )
            market = load_sample()
            source_note = "Packaged fallback dataset because live retrieval was unavailable."

    validate_history_length(market, minimum_rows=look_back + 30)
    features = create_market_features(market)
    forecast = forecast_future(
        market_df=market,
        model=model,
        scaler=scaler,
        look_back=look_back,
        horizon=int(horizon),
    )
except Exception as exc:
    st.error(f"The selected data could not be processed: {exc}")
    st.stop()

last_close = float(market["Close"].iloc[-1])
ending_forecast = float(forecast["Predicted_Close"].iloc[-1])
forecast_change = (ending_forecast / last_close - 1.0) * 100.0
direction = "Upward" if forecast_change > 0.5 else "Downward" if forecast_change < -0.5 else "Relatively flat"

summary_columns = st.columns(4)
summary_columns[0].metric("Latest Close", currency(last_close))
summary_columns[1].metric(f"{horizon}-Day Forecast", currency(ending_forecast))
summary_columns[2].metric("Forecast Change", f"{forecast_change:+.2f}%")
summary_columns[3].metric("Trend Interpretation", direction)

st.caption(source_note)

overview_tab, analysis_tab, forecast_tab, performance_tab, methodology_tab = st.tabs(
    [
        "Overview",
        "Market Analysis",
        "Future Forecast",
        "Model Performance",
        "Methodology & Limitations",
    ]
)

with overview_tab:
    st.plotly_chart(historical_figure(market), use_container_width=True)
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Prepared Data Preview")
        st.dataframe(market.tail(12), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Project Scope")
        st.markdown(
            f"""
            - **Target:** daily Bitcoin closing price in USD
            - **Input window:** previous {look_back} daily feature rows
            - **Forecast:** recursive 1–30 day demonstration
            - **Model:** stacked LSTM with 90,305 trainable parameters
            - **Inference:** pretrained weights loaded without retraining
            """
        )
        st.download_button(
            "Download prepared market data",
            data=market.to_csv(index=False).encode("utf-8"),
            file_name="prepared_bitcoin_market_data.csv",
            mime="text/csv",
        )

with analysis_tab:
    st.plotly_chart(moving_average_figure(features), use_container_width=True)
    chart_col, metric_col = st.columns([1.4, 1])
    with chart_col:
        returns_fig = go.Figure(
            go.Scatter(
                x=features["Date"],
                y=features["Return"] * 100,
                mode="lines",
                name="Daily Return",
            )
        )
        returns_fig.update_layout(
            title="Daily Bitcoin Return",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=420,
        )
        st.plotly_chart(returns_fig, use_container_width=True)
    with metric_col:
        st.subheader("Volatility Summary")
        st.metric("Average Daily Return", f"{features['Return'].mean() * 100:.3f}%")
        st.metric("Daily Return Volatility", f"{features['Return'].std() * 100:.3f}%")
        st.metric(
            "Largest Daily Gain",
            f"{features['Return'].max() * 100:.2f}%",
        )
        st.metric(
            "Largest Daily Decline",
            f"{features['Return'].min() * 100:.2f}%",
        )
        st.caption(
            "These statistics describe the selected history; they do not imply "
            "future returns or investment opportunity."
        )

with forecast_tab:
    st.plotly_chart(forecast_figure(market, forecast), use_container_width=True)
    st.subheader("Forecast Table")
    display_forecast = forecast.copy()
    display_forecast["Predicted_Close"] = display_forecast["Predicted_Close"].map(currency)
    display_forecast["Change_From_Previous_Day_Percent"] = display_forecast[
        "Change_From_Previous_Day_Percent"
    ].map(lambda value: f"{value:+.3f}%")
    st.dataframe(display_forecast, use_container_width=True, hide_index=True)

    st.download_button(
        "Download forecast CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name=f"bitcoin_lstm_forecast_{horizon}_days.csv",
        mime="text/csv",
    )
    st.info(
        f"The recursive forecast ends at {currency(ending_forecast)}, a "
        f"{forecast_change:+.2f}% change from the latest selected close. "
        "Longer recursive horizons accumulate uncertainty and should be treated "
        "as model demonstrations rather than expected investment outcomes."
    )

with performance_tab:
    st.subheader("Supplied Notebook Metrics")
    supplied_metrics = metadata.get("supplied_notebook_metrics", {})
    metric_columns = st.columns(4)
    metric_columns[0].metric("MAE", currency(float(supplied_metrics.get("MAE", 0))))
    metric_columns[1].metric("RMSE", currency(float(supplied_metrics.get("RMSE", 0))))
    metric_columns[2].metric("R²", f"{float(supplied_metrics.get('R2', 0)):.4f}")
    metric_columns[3].metric(
        "MAPE",
        "Not reported" if supplied_metrics.get("MAPE") is None else f"{supplied_metrics['MAPE']:.2f}%",
    )
    st.caption(
        "These metrics were reported by the supplied notebook on its chronological "
        "20% holdout. That same holdout was also used for validation, so the values "
        "should be treated as supplied-model results rather than a strict untouched-test estimate."
    )

    try:
        replay = replay_predictions(market, model, scaler, look_back)
        replay_view = replay.tail(max(60, int(len(replay) * 0.20))).reset_index(drop=True)
        replay_metrics = regression_metrics(
            replay_view["Actual_Close"], replay_view["Predicted_Close"]
        )
        st.subheader("Replay Metrics on the Selected Dataset")
        replay_columns = st.columns(4)
        replay_columns[0].metric("MAE", currency(replay_metrics["MAE"]))
        replay_columns[1].metric("RMSE", currency(replay_metrics["RMSE"]))
        replay_columns[2].metric("MAPE", f"{replay_metrics['MAPE']:.2f}%")
        replay_columns[3].metric("R²", f"{replay_metrics['R2']:.4f}")
        st.plotly_chart(prediction_figure(replay_view), use_container_width=True)
        st.caption(
            "Replay metrics describe one-step predictions on the selected data. "
            "For uploaded or optional recent data, they are diagnostic values and "
            "not an independent retraining result."
        )
    except Exception as exc:
        st.warning(f"Replay evaluation could not be generated: {exc}")

with methodology_tab:
    st.subheader("Model Architecture")
    st.code(
        """30 daily timesteps × 5 features
→ LSTM (128 units, return sequences)
→ Dropout (20%)
→ LSTM (32 units)
→ Dense (32 units, ReLU)
→ Dense (1 predicted scaled close)""",
        language=None,
    )

    st.subheader("Important Limitations")
    st.markdown(
        """
        - Bitcoin is highly volatile and affected by information that is not present in historical OHLCV data.
        - The packaged model was supplied with a scaler fitted before the original split.
        - The original notebook reused the holdout period for validation and final reporting.
        - Recursive forecasts reuse earlier predictions, so uncertainty compounds across days.
        - The model does not include macroeconomic, regulatory, on-chain, sentiment, order-book, or derivatives data.
        - The packaged offline sample is a deterministic demonstration dataset, not a current market feed.
        - Optional recent data may fall outside the model's original 2018–2024 training distribution.
        """
    )

    st.subheader("Responsible Interpretation")
    st.warning(FINANCIAL_DISCLAIMER)
