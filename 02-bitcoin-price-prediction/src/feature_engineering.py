from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["Close", "SMA_7", "SMA_30", "Return", "Volume"]


def create_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the exact five-feature order used by the supplied LSTM model."""
    result = df.copy()
    result["SMA_7"] = result["Close"].rolling(7, min_periods=7).mean()
    result["SMA_30"] = result["Close"].rolling(30, min_periods=30).mean()
    result["Return"] = result["Close"].pct_change()
    result["Price_Range"] = result["High"] - result["Low"]
    result["Open_Close_Difference"] = result["Close"] - result["Open"]
    result["Rolling_Volatility_14"] = result["Return"].rolling(14).std()
    result["Volume_Change"] = result["Volume"].pct_change()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def latest_model_matrix(feature_df: pd.DataFrame) -> np.ndarray:
    return feature_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
