"""End-to-end artifact-backed traffic forecasting pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data_preprocessing import prepare_traffic_data
from .feature_engineering import add_time_features
from .model_evaluation import comparison_table
from .model_evaluation import regression_metrics
from .model_evaluation import residual_frame
from .portable_model import PortableStackedLSTM
from .sequence_generation import build_sequences
from .sequence_generation import sequence_target_timestamps
from .traffic_preprocessing import ScalingArtifacts


class TrafficForecastingPipeline:
    """Load saved artifacts and produce backtests or future forecasts."""

    def __init__(
        self,
        model: PortableStackedLSTM,
        scaling: ScalingArtifacts,
        metadata: dict,
    ):
        self.model = model
        self.scaling = scaling
        self.metadata = metadata

    @classmethod
    def from_artifacts(
        cls,
        model_directory: str | Path,
    ) -> "TrafficForecastingPipeline":
        directory = Path(model_directory)
        model = PortableStackedLSTM(
            directory / "stacked_lstm_traffic.keras"
        )
        scaling = ScalingArtifacts.from_json(directory / "scalers.json")
        metadata_path = directory / "model_metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        return cls(model=model, scaling=scaling, metadata=metadata)

    def predict_next(self, frame: pd.DataFrame) -> float:
        """Predict the congestion index for the row after the latest history."""
        prepared = prepare_traffic_data(frame)
        length = self.scaling.sequence_length
        if len(prepared) < length:
            raise ValueError(
                f"At least {length} chronological rows are required."
            )
        scaled = self.scaling.transform_features(prepared.tail(length))
        prediction_scaled = self.model.predict(
            scaled.reshape(1, length, -1)
        )
        return float(self.scaling.inverse_target(prediction_scaled)[0])

    def backtest(self, frame: pd.DataFrame) -> dict:
        """Evaluate one-step forecasts over a chronological dataset."""
        prepared = prepare_traffic_data(frame)
        features = self.scaling.transform_features(prepared)
        target_scaled = self.scaling.transform_target(
            prepared["congestion_index"].to_numpy()
        )
        sequences, labels_scaled = build_sequences(
            features,
            target_scaled,
            self.scaling.sequence_length,
        )
        predictions_scaled = self.model.predict(sequences)
        actual = self.scaling.inverse_target(labels_scaled)
        predicted = self.scaling.inverse_target(predictions_scaled)
        persistence = self.scaling.inverse_target(sequences[:, -1, 0])
        timestamps = sequence_target_timestamps(
            prepared["timestamp"],
            self.scaling.sequence_length,
        )

        model_metrics = regression_metrics(actual, predicted)
        baseline_metrics = regression_metrics(actual, persistence)
        predictions = residual_frame(timestamps, actual, predicted)
        predictions["persistence_prediction"] = persistence

        return {
            "prepared_data": prepared,
            "predictions": predictions,
            "model_metrics": model_metrics,
            "baseline_metrics": baseline_metrics,
            "comparison": comparison_table(
                model_metrics,
                baseline_metrics,
            ),
        }

    @staticmethod
    def congestion_label(value: float) -> str:
        """Translate the synthetic congestion index into readable bands."""
        if value < 35:
            return "Low / off-peak"
        if value < 60:
            return "Moderate"
        return "High / peak-period"

    @staticmethod
    def _infer_frequency(timestamps: pd.Series) -> pd.Timedelta:
        differences = timestamps.sort_values().diff().dropna()
        if differences.empty:
            return pd.Timedelta(hours=1)
        frequency = differences.median()
        return (
            frequency
            if frequency > pd.Timedelta(0)
            else pd.Timedelta(hours=1)
        )

    @staticmethod
    def _seasonal_value(
        history: pd.DataFrame,
        column: str,
        future_timestamp: pd.Timestamp,
    ) -> float:
        same_slot = history[
            (history["timestamp"].dt.hour == future_timestamp.hour)
            & (
                history["timestamp"].dt.dayofweek
                == future_timestamp.dayofweek
            )
        ]
        if not same_slot.empty:
            return float(same_slot[column].tail(8).mean())
        return float(history[column].tail(24 * 7).median())

    def recursive_forecast(
        self,
        frame: pd.DataFrame,
        horizon: int = 24,
    ) -> pd.DataFrame:
        """Create a scenario-based recursive forecast for 1–24 future rows."""
        if horizon < 1 or horizon > 24:
            raise ValueError("horizon must be between 1 and 24.")

        history = prepare_traffic_data(frame)
        frequency = self._infer_frequency(history["timestamp"])
        forecasts = []

        for step in range(1, horizon + 1):
            predicted = self.predict_next(history)
            future_timestamp = history["timestamp"].iloc[-1] + frequency
            next_row = {
                "timestamp": future_timestamp,
                "vehicle_count": self._seasonal_value(
                    history,
                    "vehicle_count",
                    future_timestamp,
                ),
                "avg_speed": self._seasonal_value(
                    history,
                    "avg_speed",
                    future_timestamp,
                ),
                "occupancy": self._seasonal_value(
                    history,
                    "occupancy",
                    future_timestamp,
                ),
                "weather_severity": self._seasonal_value(
                    history,
                    "weather_severity",
                    future_timestamp,
                ),
                "congestion_index": predicted,
            }
            history = pd.concat(
                [history, add_time_features(pd.DataFrame([next_row]))],
                ignore_index=True,
            )
            forecasts.append(
                {
                    "forecast_step": step,
                    "timestamp": future_timestamp,
                    "predicted_congestion_index": predicted,
                    "traffic_band": self.congestion_label(predicted),
                    "scenario_note": (
                        "External traffic and weather inputs use recent "
                        "hour-of-week seasonal profiles."
                    ),
                }
            )
        return pd.DataFrame(forecasts)
