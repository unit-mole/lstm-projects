# Monorepo Integration

## Final location

Copy the project folder to:

```text
lstm-projects/10-traffic-flow-prediction-stacked-lstm/
```

Copy the workflow to:

```text
lstm-projects/.github/workflows/10-traffic-flow-prediction-stacked-lstm.yml
```

Do not place the workflow inside the individual project folder.

## Expected root structure

```text
lstm-projects/
├── .github/
│   └── workflows/
│       ├── 01-airline-passenger-forecasting.yml
│       ├── 02-bitcoin-price-prediction.yml
│       ├── ...
│       ├── 09-video-frame-prediction-convlstm.yml
│       └── 10-traffic-flow-prediction-stacked-lstm.yml
├── 01-airline-passenger-forecasting/
├── ...
├── 09-video-frame-prediction-convlstm/
├── 10-traffic-flow-prediction-stacked-lstm/
├── .gitignore
├── LICENSE
└── README.md
```

## Root README project entry

Add this row to the completed-projects table:

```markdown
| 10 | Traffic Flow Prediction using Stacked LSTM | Multivariate hourly traffic forecasting, baseline comparison, residual analysis, and Streamlit scenario forecasts | [Project](10-traffic-flow-prediction-stacked-lstm) | Deployment pending |
```

Add this bullet to the completed-projects section:

```markdown
- **Traffic Flow Prediction using Stacked LSTM** — Forecasts next-hour
  congestion from 24 hours of traffic, speed, occupancy, weather, and
  cyclical time features.
```

## Recommended GitHub repository description

```text
Professional LSTM portfolio featuring forecasting, anomaly detection, NLP,
attention, ConvLSTM, predictive maintenance, and deployable Streamlit apps.
```

## Recommended repository topics

```text
lstm
deep-learning
time-series
traffic-forecasting
transportation-analytics
predictive-analytics
tensorflow
keras
streamlit
machine-learning
data-science
portfolio
```

## Git commands

Run from the root `lstm-projects` folder:

```bash
git add 10-traffic-flow-prediction-stacked-lstm
git add .github/workflows/10-traffic-flow-prediction-stacked-lstm.yml
git commit -m "Add traffic flow prediction Stacked LSTM project"
git pull --rebase origin main
git push origin main
```
