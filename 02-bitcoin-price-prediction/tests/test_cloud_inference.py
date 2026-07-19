from __future__ import annotations

import numpy as np

from src.cloud_inference import NumpyBitcoinLSTM
from src.config import WEIGHTS_PATH


def test_numpy_lstm_output_shape_and_determinism() -> None:
    model = NumpyBitcoinLSTM(WEIGHTS_PATH)
    values = np.linspace(0.0, 1.0, 30 * 5, dtype=np.float32).reshape(1, 30, 5)

    first = model.predict(values)
    second = model.predict(values)

    assert first.shape == (1, 1)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_numpy_lstm_accepts_single_sequence() -> None:
    model = NumpyBitcoinLSTM(WEIGHTS_PATH)
    sequence = np.zeros((30, 5), dtype=np.float32)
    result = model.predict(sequence)

    assert result.shape == (1, 1)
