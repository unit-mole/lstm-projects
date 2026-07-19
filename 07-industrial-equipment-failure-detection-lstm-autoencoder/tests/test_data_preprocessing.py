import numpy as np
import pandas as pd

from src.data_preprocessing import DatasetSchema, clean_sensor_data


def test_cleaning_sorts_deduplicates_and_fills_sensor_values():
    frame = pd.DataFrame({
        "unit_id": [1, 1, 1, 1],
        "cycle": [3, 1, 2, 2],
        "failure_label": [0, 0, 0, 0],
        "sensor_1": [3.0, 1.0, np.nan, 2.0],
    })
    clean = clean_sensor_data(frame, DatasetSchema(sensor_cols=["sensor_1"]))
    assert clean["cycle"].tolist() == [1, 2, 3]
    assert clean["sensor_1"].tolist() == [1.0, 2.0, 3.0]
