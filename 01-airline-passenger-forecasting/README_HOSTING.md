# Hosting Guide — Streamlit Community Cloud

## Recommended platform

Use **Streamlit Community Cloud** because the application is already written in Streamlit and loads compact, pre-trained model artifacts directly from the repository.

## Files required for deployment

Keep the complete project folder committed:

```text
01-airline-passenger-forecasting/
├── app/streamlit_app.py
├── data/airline_passengers_sample.csv
├── models/airline_passenger_lstm.keras
├── models/seasonal_growth_scaler.pkl
├── models/model_metadata.json
├── outputs/test_predictions.csv
├── src/
├── requirements.txt
└── .streamlit/config.toml
```

The deployed app performs inference only. It does not retrain the model during startup.

## Deployment steps

1. Push the complete `lstm-projects` repository to the `main` branch on GitHub.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Create a new app and select the repository.
4. Use these values:

```text
Repository: <your-github-username>/lstm-projects
Branch: main
Main file path: 01-airline-passenger-forecasting/app/streamlit_app.py
Python version: 3.12
```

5. Deploy and review the build logs.
6. Test the sample dataset, CSV upload, horizon selector, charts, and forecast download.
7. Add the final app URL to the project and repository READMEs.

## Troubleshooting

- **Dependency build failure:** confirm `01-airline-passenger-forecasting/requirements.txt` is committed and Python 3.12 is selected.
- **Model not found:** confirm all required files under `01-airline-passenger-forecasting/models/` are tracked in Git.
- **Import error:** confirm the selected main file path is `01-airline-passenger-forecasting/app/streamlit_app.py`.
- **Memory pressure:** do not train the model in the hosted app; use the packaged inference artifacts.

## README badge

After deployment, replace the placeholder URL in the README badge:

```markdown
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-SUBDOMAIN.streamlit.app)
```
