# Improvement Roadmap

## High priority

1. Validate the pipeline on a licensed real-world traffic dataset.
2. Train a direct multi-horizon model for 6-, 12-, and 24-hour forecasts.
3. Add probabilistic prediction intervals.
4. Add incident, event, holiday, construction, and weather feeds.
5. Evaluate rush-hour errors with business-cost weighting.

## Modeling

- Compare persistence, seasonal naive, moving average, linear regression,
  gradient boosting, single-layer LSTM, GRU, TCN, and Transformer baselines.
- Tune sequence length, recurrent units, dropout, batch size, and learning
  rate with time-series cross-validation.
- Explore location-specific and graph-based traffic forecasting.
- Add quantile loss and calibrated uncertainty.
- Monitor recursive forecast error growth by horizon.

## Engineering

- Add experiment tracking and model registry.
- Add a FastAPI inference service.
- Add scheduled drift and data-quality checks.
- Add Docker image publishing and deployment smoke tests.
- Add model-card automation and release tagging.

## Application

- Add road or sensor selection for multi-location data.
- Add map-based visualizations for licensed geospatial data.
- Add downloadable PDF/HTML forecast reports.
- Add configurable congestion thresholds.
- Add comparison between saved model versions.

## Responsible deployment

- Establish operational owners and escalation procedures.
- Define forecast-confidence requirements.
- Validate across seasons, incidents, holidays, and weather extremes.
- Document safety boundaries and human-review requirements.
