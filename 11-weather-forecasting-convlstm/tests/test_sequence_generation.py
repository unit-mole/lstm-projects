import numpy as np

from src.sequence_generation import create_rolling_sequences


def test_rolling_sequences_preserve_order():
    frames = np.arange(10 * 4 * 4, dtype="float32").reshape(10, 4, 4, 1)
    X, y = create_rolling_sequences(frames, input_frames=3, forecast_horizon=1)
    assert X.shape == (7, 3, 4, 4, 1)
    assert y.shape == (7, 4, 4, 1)
    np.testing.assert_array_equal(X[0], frames[:3])
    np.testing.assert_array_equal(y[0], frames[3])
