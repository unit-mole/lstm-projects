import numpy as np
import pytest

from src.prediction_pipeline import validate_sequence_shape


def test_validate_sequence_shape_accepts_expected_shape():
    sequence = np.zeros((6, 32, 32, 1), dtype=np.float32)
    result = validate_sequence_shape(sequence)
    assert result.shape == sequence.shape


def test_validate_sequence_shape_rejects_wrong_length():
    with pytest.raises(ValueError):
        validate_sequence_shape(np.zeros((5, 32, 32, 1), dtype=np.float32))
