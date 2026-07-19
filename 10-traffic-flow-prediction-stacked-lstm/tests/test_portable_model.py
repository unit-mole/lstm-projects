from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preprocessing import prepare_traffic_data
from src.portable_model import PortableStackedLSTM
from src.sequence_generation import build_sequences
from src.traffic_preprocessing import ScalingArtifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_produces_finite_predictions():
    scaling = ScalingArtifacts.from_json(
        PROJECT_ROOT / "models" / "scalers.json"
    )
    model = PortableStackedLSTM(
        PROJECT_ROOT / "models" / "stacked_lstm_traffic.keras"
    )
    frame = prepare_traffic_data(
        pd.read_csv(
            PROJECT_ROOT / "data" / "sample_traffic_flow_data.csv"
        )
    )
    features = scaling.transform_features(frame)
    targets = scaling.transform_target(
        frame["congestion_index"].to_numpy()
    )
    X, _ = build_sequences(
        features,
        targets,
        scaling.sequence_length,
    )
    predictions = model.predict(X[:3])

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
