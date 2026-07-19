# Project Audit

## Current objective identified

- **Task:** multivariate hourly time-series regression.
- **Target:** `energy_load`.
- **Inputs:** historical load, temperature, humidity, cyclical hour/day features and weekend indicator.
- **Forecast setup:** previous 24 hours predict the next hour.
- **Model:** three stacked LSTM layers followed by dense regression layers.
- **Business framing:** short-horizon demand forecasting for capacity and operational planning.

## Data audit

- Records: **17,520**.
- Frequency: **hourly**.
- Range: **2022-01-01 00:00:00 to 2023-12-31 23:00:00**.
- Missing values in the supplied synthetic data: **0**.
- Duplicate timestamps: **0**.
- Data source: reproducible synthetic fallback generated in the original notebook.

## Leakage controls confirmed

- Chronological 70/15/15 split.
- No random train/test shuffle.
- Feature and target scalers fitted on training rows only.
- Input windows use past rows only.
- Test metrics are calculated on the future unseen partition.

## Artifact verification

The supplied Keras weights were independently evaluated against the reproducible dataset and scaler metadata. Results match the notebook:

| Model | Test MAE | Test RMSE | Test MAPE | Test R² |
|---|---:|---:|---:|---:|
| Naive previous value | 8.5287 | 10.8063 | 8.72% | 0.7558 |
| Stacked LSTM | 5.0277 | 6.3226 | 5.09% | 0.9164 |

## Remaining production gaps

- Synthetic training data limits real-world generalization claims.
- No prediction interval or uncertainty calibration is included.
- Recursive forecasts accumulate error.
- Exogenous weather inputs must be known or forecast separately.
- Backtesting across multiple rolling origins should be added before production use.

## Monorepo CI integration

- Workflow path: `.github/workflows/08-multivariate-time-series-forecasting-stacked-lstm.yml`
- Trigger scope: changes to Project 08 or its workflow only.
- Python version: 3.11.
- Checks: syntax compilation, critical Ruff rules, Pytest suite, required-artifact validation, and notebook-format validation.
- CI uses lightweight test dependencies and does not retrain or load the TensorFlow model, keeping pull-request checks faster and more reliable.
