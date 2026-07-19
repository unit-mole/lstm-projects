import numpy as np

from src.synthetic_weather import generate_weather_sequences


def test_generator_shape_and_range():
    X, future = generate_weather_sequences(n_samples=3, input_frames=6, future_frames=2, grid_size=24, seed=7)
    assert X.shape == (3, 6, 24, 24, 1)
    assert future.shape == (3, 2, 24, 24, 1)
    assert X.dtype == np.float32
    assert 0.0 <= float(X.min()) <= float(X.max()) <= 1.0
