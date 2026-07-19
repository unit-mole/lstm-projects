import numpy as np

from src.model_evaluation import regression_metrics


def test_regression_metrics_are_zero_for_perfect_prediction():
    actual = np.array([10.0, 20.0, 30.0])
    metrics = regression_metrics(actual, actual.copy())
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["mape_pct"] == 0.0
    assert metrics["r2"] == 1.0
