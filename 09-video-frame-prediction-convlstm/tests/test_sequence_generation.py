import numpy as np

from src.sequence_generation import create_next_frame_sequences, generate_moving_sequences


def test_synthetic_generator_reproducible():
    x1, y1 = generate_moving_sequences(n_samples=3, seed=42)
    x2, y2 = generate_moving_sequences(n_samples=3, seed=42)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert x1.shape == (3, 6, 32, 32, 1)
    assert y1.shape == (3, 32, 32, 1)


def test_create_next_frame_sequences_preserves_order():
    frames = np.arange(8, dtype=np.float32).reshape(8, 1, 1, 1)
    x, y = create_next_frame_sequences(frames, input_frames=3)
    assert x[0, :, 0, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert y[0, 0, 0, 0] == 3.0
