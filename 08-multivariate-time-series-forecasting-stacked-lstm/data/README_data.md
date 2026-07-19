# Dataset Notes

This project uses a **reproducible synthetic hourly energy-demand dataset** because no distributable external dataset was supplied with the original notebook.

## Files

- `hourly_energy.csv`: 17,520 hourly rows from 2022-01-01 through 2023-12-31. This reproduces the model-training dataset.
- `sample_multivariate_timeseries.csv`: safe 90-day subset used by the Streamlit demo.
- `future_exogenous_sample.csv`: 24 future hourly rows containing temperature and humidity assumptions for recursive forecasting.

## Core schema

| Column | Role |
|---|---|
| `timestamp` | Hourly time index |
| `energy_load` | Forecast target and autoregressive input feature |
| `temperature` | Exogenous weather feature |
| `humidity` | Exogenous weather feature |
| `hour_sin`, `hour_cos` | Cyclical hour-of-day features |
| `dow_sin`, `dow_cos` | Cyclical day-of-week features |
| `weekend` | Weekend indicator |

The generator is available in `scripts/generate_sample_data.py`. The dataset is suitable for portfolio demonstration only and should not be interpreted as real utility consumption.
