import pandas as pd

from src.data_preprocessing import chronological_split, prepare_time_series


def test_prepare_time_series_sorts_and_deduplicates():
    frame = pd.DataFrame({
        "timestamp": ["2024-01-02", "2024-01-01", "2024-01-01"],
        "energy_load": [3.0, 1.0, 2.0],
        "temperature": [20.0, 19.0, 19.5],
        "humidity": [50.0, 51.0, 52.0],
    })
    result, report = prepare_time_series(frame)
    assert list(result["energy_load"]) == [2.0, 3.0]
    assert report.duplicate_timestamps_removed == 1


def test_chronological_split_preserves_order():
    frame = pd.DataFrame({"value": range(20)})
    train, validation, test = chronological_split(frame, 0.6, 0.2)
    assert train["value"].tolist() == list(range(12))
    assert validation["value"].tolist() == list(range(12, 16))
    assert test["value"].tolist() == list(range(16, 20))
