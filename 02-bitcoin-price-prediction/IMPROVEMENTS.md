# Review of the Original Bitcoin Notebook

## What the original project did well

The supplied notebook already contained a meaningful end-to-end foundation:

- Yahoo Finance `BTC-USD` download from 2018 through 2024;
- daily OHLCV inspection and chronological ordering;
- 7-day and 30-day moving averages;
- daily returns and volume as model features;
- multivariate 30-day LSTM sequences;
- a manual architecture search;
- a stacked LSTM with dropout;
- early stopping and learning-rate reduction;
- model saving, scaler saving, and artifact reload verification;
- one-step holdout predictions and a recursive ten-day forecast.

The supplied model reported:

```text
RMSE: 2,426.9979 USD
MAE:  1,651.0163 USD
R²:   0.9845
```

## Methodological issues identified

### 1. Scaler leakage

The original notebook fitted `MinMaxScaler` on the complete feature matrix before splitting the
sequences. Future validation and holdout values therefore influenced the scale parameters.

The cleaned retraining pipeline fits the scaler only on training-period feature rows.

### 2. Validation and test reuse

The original final model used the last 20% of sequences as `validation_data` for early stopping and
then reported final metrics on that same period. This makes the reported results less independent
than a strict untouched test.

The cleaned pipeline creates chronological 70% training, 15% validation, and 15% test periods.

### 3. Hyperparameter search used a future holdout

Each candidate architecture was compared on the final 20% of the series, and the same period was
later reused. The cleaned design keeps validation-based selection separate from final testing.

### 4. Recursive feature update

The original multi-step forecast changed only the scaled close feature while leaving moving averages,
return, and volume unchanged. The revised inference pipeline recalculates:

- 7-day moving average;
- 30-day moving average;
- daily return;
- recent-volume carry-forward value.

### 5. Legacy model format

The original model was saved as `.h5`. The portfolio version preserves the original artifact in the
archive and supplies a native `.keras` version.

### 6. Cloud-runtime reliability

A lightweight `.npz` weight artifact and NumPy forward pass were added so Streamlit Cloud does not
need to start a deep-learning backend during inference.

### 7. Missing portfolio engineering

The rebuilt project adds:

- modular source files;
- configurable data loading;
- robust CSV standardization;
- optional recent-data retrieval with fallback;
- explicit financial disclaimer;
- baseline utilities;
- residual and volatility analysis;
- downloadable forecasts;
- tests and GitHub Actions;
- local and hosting instructions;
- recruiter-facing documentation.

## Honest interpretation

A high R² for price-level prediction does not prove trading usefulness. Bitcoin prices are highly
persistent, so a previous-close baseline can also achieve strong price-level metrics. A production
study should compare return forecasts, directional accuracy, transaction costs, uncertainty, and
walk-forward performance before making any claim of predictive edge.
