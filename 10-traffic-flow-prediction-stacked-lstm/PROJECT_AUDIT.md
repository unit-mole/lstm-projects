# Project Audit

## Supplied assets reviewed

- `Traffic_Flow_Prediction_Deep_Stacked_LSTM_FULL_ELITE.ipynb`
- `stacked_lstm_traffic.keras`
- `scalers.json`

## What the original project does

The supplied notebook creates a deterministic hourly traffic dataset with
commute peaks, weekend effects, speed, occupancy, weather severity, and a
continuous congestion index. It uses the previous 24 hours of 10 scaled
features to predict the next congestion index with a three-layer Stacked
LSTM.

## Verified artifact configuration

- Sequence length: 24
- Feature count: 10
- Target: `congestion_index`
- Architecture: LSTM 64 → LSTM 32 → LSTM 16 → Dense 16 → Dense 1
- Trainable parameters: 35,041
- Chronological split: 70% / 15% / 15%
- Test MAE: 2.691659
- Test RMSE: 3.379029
- Test R²: 0.950879

The portable NumPy inference implementation reproduces the supplied model's
held-out metrics and first predictions within floating-point tolerance.

## Strengths retained

- Correct chronological split
- Training-only scaling
- Multivariate 24-hour windows
- Stacked recurrent architecture
- Persistence baseline
- Validation and test evaluation
- Residual analysis
- Saved Keras and scaler artifacts

## Issues addressed

- Replaced notebook-only execution with reusable modules.
- Removed machine-specific Windows paths.
- Added robust timestamp, duplicate, and missing-value handling.
- Added schema validation for uploads.
- Added portable inference so Streamlit does not require TensorFlow.
- Added model metadata and artifact hashes.
- Added a safe GitHub sample dataset.
- Added a polished Streamlit application.
- Added recursive scenario forecasting with explicit limitations.
- Added pytest, Ruff, compilation, and artifact smoke checks.
- Added root-level monorepo GitHub Actions integration.
- Added hosting, audit, improvement, and data documentation.
- Added a cleaned notebook that uses the project modules.
- Added reproducible output plots and evaluation tables.

## Important technical note

The supplied model is a direct one-step forecaster. The application's
multi-step view is recursive and uses hour-of-week profiles for unknown
external inputs. This is stated clearly to avoid overstating model
capability.

## Data-safety assessment

The original executed dataset is synthetic. The included sample contains no
proprietary traffic infrastructure or personally identifiable information.
