import pandas as pd

from src.data_preprocessing import DatasetSchema
from src.sequence_generation import build_sequences


def test_windows_do_not_cross_equipment_ids():
    frame = pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2, 2],
        "cycle": [1, 2, 3, 1, 2, 3],
        "failure_label": [0, 0, 1, 0, 0, 1],
        "sensor_1": [1, 2, 3, 10, 11, 12],
    })
    batch = build_sequences(
        frame, DatasetSchema(sensor_cols=["sensor_1"]), window_size=2
    )
    assert batch.sequences.shape == (4, 2, 1)
    assert batch.unit_ids.tolist() == [1, 1, 2, 2]
    assert batch.labels.tolist() == [0, 1, 0, 1]
