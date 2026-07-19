from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import prepare_ecg_frame
from src.signal_preprocessing import interpolate_signal


def test_prepare_ecg_frame_standardizes_columns() -> None:
    values = np.linspace(0.0, 1.0, 140)
    frame = pd.DataFrame(
        [values, values],
        columns=[f"value_{index}" for index in range(140)],
    )
    frame.insert(0, "class", [0, 0])

    prepared = prepare_ecg_frame(frame)

    assert len(prepared) == 1
    assert prepared.loc[0, "label"] == 0
    assert "sample_139" in prepared.columns


def test_interpolate_signal_returns_required_length() -> None:
    signal = np.asarray([0.0, 1.0, 0.0])
    result = interpolate_signal(signal, sequence_length=140)

    assert result.shape == (140,)
    assert np.isfinite(result).all()


def test_prepare_ecg_frame_rejects_wrong_shape() -> None:
    frame = pd.DataFrame({"sample_000": [0.1], "sample_001": [0.2]})
    with pytest.raises(ValueError):
        prepare_ecg_frame(frame)
