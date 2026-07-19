import numpy as np

from src.sequence_generation import build_sequences


def test_sequence_shape_and_target_alignment():
    features = np.arange(60, dtype=float).reshape(20, 3)
    targets = np.arange(20, dtype=float)

    X, y = build_sequences(features, targets, sequence_length=5)

    assert X.shape == (15, 5, 3)
    assert y.shape == (15,)
    np.testing.assert_array_equal(X[0], features[:5])
    assert y[0] == targets[5]
