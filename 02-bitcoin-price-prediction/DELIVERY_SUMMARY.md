# Bitcoin Price Prediction Delivery Summary

## Supplied files reviewed

- Original Bitcoin LSTM notebook
- Selected configuration pickle
- Trained `.h5` model
- Fitted five-feature scaler

## Main findings

- Source workflow: daily Yahoo Finance `BTC-USD` OHLCV data
- Original date range: January 2018 through December 2024
- Target: `Close`
- Features: Close, SMA 7, SMA 30, daily return, and volume
- Input window: 30 days
- Architecture: stacked LSTM with 90,305 trainable parameters
- Supplied metrics: MAE $1,651.02, RMSE $2,427.00, R² 0.9845
- Original scaler was fitted before splitting
- Original holdout was reused for validation and final reporting

## Portfolio upgrades

- Complete modular source package
- Robust CSV preparation and validation
- Optional recent-data retrieval with offline fallback
- NumPy-only cloud inference artifact
- Recursive feature recalculation
- Model and baseline evaluation utilities
- Streamlit app with financial disclaimer
- Cleaned notebook
- Automated tests and GitHub Actions workflow
- Project README and hosting guide
- Main repository README update
