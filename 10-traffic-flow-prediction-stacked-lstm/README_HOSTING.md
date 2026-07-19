# Streamlit Community Cloud Deployment Guide

## Recommended deployment

Use Streamlit Community Cloud because the project already lives inside the
public `unit-mole/lstm-projects` monorepo and the app has a dedicated
deployment requirements file beside its entrypoint.

## Required GitHub paths

```text
10-traffic-flow-prediction-stacked-lstm/app/streamlit_app.py
10-traffic-flow-prediction-stacked-lstm/app/requirements.txt
10-traffic-flow-prediction-stacked-lstm/data/sample_traffic_flow_data.csv
10-traffic-flow-prediction-stacked-lstm/models/stacked_lstm_traffic.keras
10-traffic-flow-prediction-stacked-lstm/models/scalers.json
10-traffic-flow-prediction-stacked-lstm/models/model_metadata.json
10-traffic-flow-prediction-stacked-lstm/src/
```

## Deployment settings

1. Sign in to Streamlit Community Cloud with the GitHub account that can
   access `unit-mole/lstm-projects`.
2. Click **Create app**.
3. Select the existing GitHub repository.
4. Enter:

   ```text
   Repository: unit-mole/lstm-projects
   Branch: main
   Main file path: 10-traffic-flow-prediction-stacked-lstm/app/streamlit_app.py
   Python version: 3.11
   ```

5. No secrets are required.
6. Choose an available application URL.
7. Click **Deploy**.

## Local pre-deployment test

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r 10-traffic-flow-prediction-stacked-lstm/app/requirements.txt
python -m streamlit run 10-traffic-flow-prediction-stacked-lstm/app/streamlit_app.py
```

Confirm:

- The sample data loads.
- Traffic patterns render.
- The model backtest completes.
- Actual and predicted values display.
- The recursive forecast generates 1–24 rows.
- CSV downloads work.
- The responsible-use disclaimer is visible.

## After deployment

Add the public URL to:

1. This project's `README.md`
2. The root `lstm-projects/README.md`
3. Your resume, LinkedIn project section, and portfolio

Replace the README's deployment-ready Streamlit badge link with the live
application URL.

## Updating the live app

Streamlit detects changes pushed to the connected GitHub branch.

```bash
git add 10-traffic-flow-prediction-stacked-lstm
git commit -m "Update Project 10 Streamlit application"
git push origin main
```

## Common errors

### `ModuleNotFoundError: No module named 'src'`

Verify that the entrypoint is exactly:

```text
10-traffic-flow-prediction-stacked-lstm/app/streamlit_app.py
```

and that `10-traffic-flow-prediction-stacked-lstm/src/` is present in GitHub.

### Model or scaler not found

Verify the three model artifacts in `10-traffic-flow-prediction-stacked-lstm/models/` are committed
and are not excluded by `.gitignore`.

### Dependency failure

Confirm Streamlit is reading:

```text
10-traffic-flow-prediction-stacked-lstm/app/requirements.txt
```

The deployed app intentionally does not install TensorFlow. It executes the
supplied Keras model through the project's NumPy inference implementation.

### App runs but forecast fails

Uploaded data must contain:

```text
timestamp
vehicle_count
avg_speed
occupancy
weather_severity
congestion_index
```

At least 24 chronological rows are required for the next-step forecast and
at least 25 rows are required for backtesting.
