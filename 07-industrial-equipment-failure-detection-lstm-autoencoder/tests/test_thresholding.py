import numpy as np

from src.thresholding import health_status, select_threshold


def test_mean_plus_three_std_threshold():
    errors = np.array([1.0, 1.0, 1.0])
    assert select_threshold(errors) == 1.0


def test_health_bands():
    assert health_status(0.9, 1.0) == "Normal Operation"
    assert health_status(1.2, 1.0).startswith("Warning")
    assert health_status(1.6, 1.0).startswith("Potential Failure")
