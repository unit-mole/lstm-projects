from pathlib import Path

import numpy as np

from src.config import LOOKBACK, METADATA_PATH, MODEL_PATH, SAMPLE_DATA_PATH, SCALER_PATH, SEASONAL_PERIOD
from src.data_preprocessing import load_and_prepare
from src.forecasting_pipeline import NumpyLSTMInferenceModel, load_artifacts, recursive_forecast


def test_numpy_lstm_artifact_loads_and_forecasts_without_keras():
    project_root = Path(__file__).resolve().parents[1]
    weights_path = project_root / "models" / "airline_passenger_lstm_weights.npz"
    model = NumpyLSTMInferenceModel.load(weights_path)
    assert model.units == 16

    loaded_model, scaler, _ = load_artifacts(MODEL_PATH, SCALER_PATH, METADATA_PATH)
    assert isinstance(loaded_model, NumpyLSTMInferenceModel)

    frame, _ = load_and_prepare(SAMPLE_DATA_PATH)
    forecast = recursive_forecast(
        frame,
        loaded_model,
        scaler,
        horizon=12,
        lookback=LOOKBACK,
        seasonal_period=SEASONAL_PERIOD,
    )
    assert len(forecast) == 12
    assert np.isfinite(forecast["Forecasted_Passengers"]).all()
    assert (forecast["Forecasted_Passengers"] >= 0).all()
