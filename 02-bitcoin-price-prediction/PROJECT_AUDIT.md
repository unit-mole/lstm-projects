# Project Audit

## Supplied artifacts inspected

- `Bitcoin Price Prediction using LSTM Deep Learning.ipynb`
- `bitcoin_lstm_model.h5`
- `bitcoin_scaler.pkl`
- `bitcoin_best_config.pkl`

## Confirmed supplied architecture

```text
Input: 30 days × 5 features
LSTM: 128 units, return sequences
Dropout: 20%
LSTM: 32 units
Dense: 32 units, ReLU
Dense: 1 regression output
Parameters: 90,305
```

## Confirmed model features

```text
Close
SMA_7
SMA_30
Return
Volume
```

## Confirmed original dataset

```text
Ticker: BTC-USD
Source: yfinance / Yahoo Finance
Download range: 2018-01-01 through 2025-01-01
Rows downloaded: 2,557
Target: Close
```

## Confirmed original reported metrics

```text
RMSE: 2426.9979
MAE: 1651.0163
R²: 0.9845
```

## Portfolio readiness

The project is ready for GitHub and Streamlit deployment as an educational demonstration. The
supplied metrics are documented with their validation limitations. Strict retraining requires
downloading or supplying a real OHLCV dataset and running `train_model.py`.
