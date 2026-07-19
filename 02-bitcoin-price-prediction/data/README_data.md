# Bitcoin Data Notes

## Original notebook source

The supplied notebook downloaded daily `BTC-USD` market data with `yfinance` for the fixed period
from January 1, 2018 through January 1, 2025. The downloaded table contained:

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close`
- `Volume`

The original workflow selected `Close` as the forecasting target and retained `Volume` as a model
feature. For Bitcoin, Yahoo Finance's `Close` and `Adj Close` were equal throughout the inspected
sample because cryptocurrency does not have stock splits or cash dividends in the same sense as an
equity security.

## Packaged offline sample

`bitcoin_price_sample.csv` is a deterministic, synthetic OHLCV demonstration dataset with 731 daily
rows and the same input schema expected by the application. It exists so the deployed app remains
functional when an external market-data service is unavailable.

The offline sample is **not an official market history** and must not be used for financial analysis,
trading, or investment decisions. It is used only to exercise preprocessing, inference, charting,
download, and validation paths.

## Accepted upload format

The Streamlit application requires:

```text
Date,Close
```

It also accepts:

```text
Date,Open,High,Low,Close,Volume
```

Common alternatives such as `Datetime`, `Timestamp`, `Adj Close`, and `Price` are detected where
possible. If OHLC or volume fields are absent, the app creates conservative placeholders because the
packaged model uses `Close`, moving averages, returns, and volume.

## Optional recent data

The application can try to fetch recent `BTC-USD` data through `yfinance`. This is optional:

- the application does not depend on live retrieval;
- network or provider failures fall back to the packaged sample;
- recent prices may fall outside the model's original 2018–2024 training distribution;
- external data-provider terms and availability apply.

## Data safety

No personal, customer, employee, confidential, or proprietary data is included. Uploaded CSV files
are processed only during the active Streamlit session.
