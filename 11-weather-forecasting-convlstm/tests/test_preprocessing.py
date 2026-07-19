import numpy as np
import pytest

from src.weather_preprocessing import repair_weather_array, validate_sequence_shape


def test_repair_weather_array():
    values = np.array([np.nan, -1.0, 0.5, 2.0], dtype="float32")
    repaired = repair_weather_array(values)
    assert np.isfinite(repaired).all()
    assert repaired.min() >= 0
    assert repaired.max() <= 1


def test_shape_validation_rejects_wrong_input():
    with pytest.raises(ValueError):
        validate_sequence_shape(np.zeros((5, 24, 24, 1), dtype="float32"))
