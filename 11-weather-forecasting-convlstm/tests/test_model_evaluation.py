import numpy as np

from src.model_evaluation import map_mae, map_rmse, threshold_metrics


def test_map_metrics():
    actual = np.zeros((2, 2, 1), dtype="float32")
    predicted = np.ones((2, 2, 1), dtype="float32")
    assert map_mae(actual, predicted) == 1.0
    assert map_rmse(actual, predicted) == 1.0


def test_threshold_metrics_perfect_match():
    values = np.array([[0.1, 0.9], [0.2, 0.8]], dtype="float32")
    result = threshold_metrics(values, values, threshold=0.5)
    assert result["iou"] == 1.0
    assert result["pixel_accuracy"] == 1.0
    assert result["far"] == 0.0
