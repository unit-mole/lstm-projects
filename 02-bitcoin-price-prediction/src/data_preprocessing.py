from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

REQUIRED_CORE_COLUMNS = ["Date", "Close"]
OPTIONAL_MARKET_COLUMNS = ["Open", "High", "Low", "Volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance-style MultiIndex columns and strip whitespace."""
    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            "_".join(str(part) for part in col if str(part) not in {"", "None"}).strip("_")
            for col in result.columns
        ]
    result.columns = [str(col).strip() for col in result.columns]
    return result


def _resolve_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {col.lower().replace("_", " ").strip(): col for col in columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", " ").strip()
        if key in normalized:
            return normalized[key]
    for col in columns:
        col_key = col.lower().replace("_", " ")
        if any(candidate.lower() in col_key for candidate in candidates):
            return col
    return None


def load_csv(source: str | Path | BinaryIO | BytesIO) -> pd.DataFrame:
    """Load a CSV from a path or uploaded file object."""
    try:
        return pd.read_csv(source)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="latin-1")


def standardize_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Bitcoin CSV column names to Date/Open/High/Low/Close/Volume."""
    result = _flatten_columns(df)

    # Use an existing index as Date when appropriate.
    if not any(str(c).lower() in {"date", "datetime", "timestamp"} for c in result.columns):
        if isinstance(result.index, pd.DatetimeIndex):
            result = result.reset_index().rename(columns={result.index.name or "index": "Date"})

    mapping: dict[str, str] = {}
    candidates = {
        "Date": ["date", "datetime", "timestamp", "time"],
        "Open": ["open"],
        "High": ["high"],
        "Low": ["low"],
        "Close": ["close", "adj close", "adjusted close", "price"],
        "Volume": ["volume", "volume usd", "volume_usd"],
    }
    for standard, options in candidates.items():
        found = _resolve_column(list(result.columns), options)
        if found is not None:
            mapping[found] = standard

    result = result.rename(columns=mapping)
    missing = [col for col in REQUIRED_CORE_COLUMNS if col not in result.columns]
    if missing:
        raise ValueError(
            "The CSV must contain a date column and a closing-price column. "
            f"Missing standardized column(s): {', '.join(missing)}."
        )

    return result


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare daily Bitcoin data while preserving chronological order.

    Only Date and Close are mandatory. Missing OHLC values are inferred from
    Close because the packaged model uses Close-derived features and Volume.
    """
    result = standardize_market_columns(df)
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in result.columns]
    result = result[keep].copy()

    result["Date"] = pd.to_datetime(result["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    for col in [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in result.columns]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.dropna(subset=["Date", "Close"])
    result = result.sort_values("Date")
    result = result.groupby("Date", as_index=False).last()

    if (result["Close"] <= 0).any():
        raise ValueError("Closing prices must be positive.")

    # Reindex to a continuous daily calendar because Bitcoin trades every day.
    full_dates = pd.date_range(result["Date"].min(), result["Date"].max(), freq="D")
    result = result.set_index("Date").reindex(full_dates)
    result.index.name = "Date"
    result["Close"] = result["Close"].interpolate("time").ffill().bfill()

    if "Open" not in result:
        result["Open"] = result["Close"].shift(1)
    result["Open"] = result["Open"].interpolate("time").fillna(result["Close"])

    if "High" not in result:
        result["High"] = np.maximum(result["Open"], result["Close"])
    result["High"] = result["High"].interpolate("time")
    result["High"] = np.maximum.reduce(
        [result["High"].to_numpy(), result["Open"].to_numpy(), result["Close"].to_numpy()]
    )

    if "Low" not in result:
        result["Low"] = np.minimum(result["Open"], result["Close"])
    result["Low"] = result["Low"].interpolate("time")
    result["Low"] = np.minimum.reduce(
        [result["Low"].to_numpy(), result["Open"].to_numpy(), result["Close"].to_numpy()]
    )

    if "Volume" not in result:
        result["Volume"] = 30_000_000_000.0
    result["Volume"] = result["Volume"].interpolate("time").ffill().bfill()
    result["Volume"] = result["Volume"].clip(lower=0)

    return result.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]


def validate_history_length(df: pd.DataFrame, minimum_rows: int = 60) -> None:
    if len(df) < minimum_rows:
        raise ValueError(
            f"At least {minimum_rows} daily rows are required; the prepared dataset contains {len(df)}."
        )


def fetch_optional_yfinance(
    ticker: str = "BTC-USD",
    period: str = "2y",
) -> pd.DataFrame:
    """
    Fetch optional recent data. This function is intentionally isolated so the
    application can fall back to the packaged sample when internet access fails.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed in this environment.") from exc

    data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise RuntimeError("No live market data was returned.")
    data = data.reset_index()
    return clean_market_data(data)
