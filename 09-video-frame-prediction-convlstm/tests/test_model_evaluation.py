import numpy as np

from src.model_evaluation import calculate_frame_metrics


def test_perfect_prediction_metrics():
    truth = np.zeros((2, 32, 32, 1), dtype=np.float32)
    metrics = calculate_frame_metrics(truth, truth)
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["pixel_accuracy"] == 1.0
