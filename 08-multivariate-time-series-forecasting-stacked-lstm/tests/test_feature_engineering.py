import pandas as pd

from src.feature_engineering import add_calendar_features


def test_calendar_features_are_created():
    frame = pd.DataFrame({"timestamp": ["2024-01-06 03:00:00"]})
    result = add_calendar_features(frame)
    assert result.loc[0, "hour"] == 3
    assert result.loc[0, "dayofweek"] == 5
    assert result.loc[0, "weekend"] == 1
    assert {"hour_sin", "hour_cos", "dow_sin", "dow_cos"}.issubset(result.columns)
