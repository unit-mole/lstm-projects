import numpy as np

from src.sequence_generation import build_supervised_sequences


def test_sequence_shapes_for_single_step():
    features = np.arange(40, dtype=float).reshape(10, 4)
    target = np.arange(10, dtype=float)
    x, y = build_supervised_sequences(features, target, sequence_length=3, forecast_horizon=1)
    assert x.shape == (7, 3, 4)
    assert y.shape == (7,)
    np.testing.assert_array_equal(x[0], features[:3])
    assert y[0] == target[3]


def test_sequence_shapes_for_multi_step():
    features = np.arange(60, dtype=float).reshape(15, 4)
    target = np.arange(15, dtype=float)
    x, y = build_supervised_sequences(features, target, sequence_length=4, forecast_horizon=3)
    assert x.shape == (9, 4, 4)
    assert y.shape == (9, 3)
