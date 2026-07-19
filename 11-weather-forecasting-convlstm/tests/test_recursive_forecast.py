import numpy as np

from src.forecasting_pipeline import recursive_forecast


class DummyModel:
    def predict(self, batch, verbose=0):
        return batch[:, -1]


def test_recursive_forecast_shape():
    sequence = np.zeros((6, 24, 24, 1), dtype="float32")
    metadata = {"input_frames": 6, "height": 24, "width": 24, "channels": 1}
    forecasts = recursive_forecast(DummyModel(), sequence, metadata, steps=3)
    assert forecasts.shape == (3, 24, 24, 1)
