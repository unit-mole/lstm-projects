from pathlib import Path

import numpy as np

from src.numpy_model import NumpyLSTMAutoencoder


def test_portable_model_output_shape():
    model_path = Path(__file__).resolve().parents[1] / "models" / "lstm_autoencoder_predictive_maintenance.keras"
    model = NumpyLSTMAutoencoder.from_keras(model_path)
    prediction = model.predict(np.zeros((2, 20, 8), dtype=np.float32))
    assert prediction.shape == (2, 20, 8)
    assert np.isfinite(prediction).all()
