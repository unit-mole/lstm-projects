# Monorepo Integration

Place the project and workflow at these exact locations:

```text
lstm-projects/
├── .github/
│   └── workflows/
│       └── 11-weather-forecasting-convlstm.yml
└── 11-weather-forecasting-convlstm/
```

## Main repository README row

Add this row to the completed-projects table in the root `README.md`:

```markdown
| 11 | [Weather Forecasting using ConvLSTM](11-weather-forecasting-convlstm/) | ConvLSTM2D, spatiotemporal grid forecasting, recursive prediction | [Live demo](STREAMLIT_URL) |
```

## Suggested root repository description update

> A professional portfolio of LSTM, attention, autoencoder, stacked-LSTM, and ConvLSTM projects covering forecasting, NLP, anomaly detection, activity recognition, video prediction, and spatiotemporal analytics.

## Git commands

```bash
git add "11-weather-forecasting-convlstm"
git add ".github/workflows/11-weather-forecasting-convlstm.yml"
git commit -m "Add ConvLSTM weather forecasting project"
git push origin main
```
