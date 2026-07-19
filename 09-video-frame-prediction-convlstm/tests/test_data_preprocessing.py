import numpy as np

from src.data_preprocessing import normalize_frame, prepare_frame_sequence


def test_normalize_uint8_frame():
    frame = np.full((8, 8), 255, dtype=np.uint8)
    result = normalize_frame(frame)
    assert result.dtype == np.float32
    assert float(result.min()) == 1.0
    assert float(result.max()) == 1.0


def test_prepare_sequence_shape():
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(6)]
    sequence = prepare_frame_sequence(frames, input_frames=6, target_size=(32, 32))
    assert sequence.shape == (6, 32, 32, 1)
