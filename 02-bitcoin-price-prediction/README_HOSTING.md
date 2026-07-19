# Streamlit Community Cloud Hosting Guide

## Recommended deployment

Streamlit Community Cloud is the recommended host because the project already includes:

- a Streamlit entry point;
- a lightweight cloud dependency file;
- a pretrained inference artifact;
- no requirement to retrain during startup;
- no mandatory API key or secret.

## GitHub structure required

The repository should contain:

```text
.github/workflows/02-bitcoin-price-prediction.yml
02-bitcoin-price-prediction/app/streamlit_app.py
02-bitcoin-price-prediction/app/requirements.txt
02-bitcoin-price-prediction/data/bitcoin_price_sample.csv
02-bitcoin-price-prediction/models/bitcoin_lstm_weights.npz
02-bitcoin-price-prediction/models/bitcoin_scaler.pkl
02-bitcoin-price-prediction/models/best_config.json
02-bitcoin-price-prediction/models/model_metadata.json
02-bitcoin-price-prediction/src/
```

## Deployment configuration

Create a new app in Streamlit Community Cloud and use:

```text
Repository: unit-mole/lstm-projects
Branch: main
Main file path: 02-bitcoin-price-prediction/app/streamlit_app.py
Python version: 3.12
```

Suggested custom subdomain:

```text
bitcoin-price-lstm
```

No secret is required. The optional `yfinance` data source uses public market-data access and the app
falls back to the packaged sample when retrieval is unavailable.

## Why the cloud app does not load Keras

The original model remains stored as:

```text
models/bitcoin_lstm_model.keras
```

For stable and lightweight deployment, the Streamlit application uses:

```text
models/bitcoin_lstm_weights.npz
```

The NumPy inference implementation reproduces the supplied two-layer LSTM forward pass without
starting TensorFlow, Keras, or JAX. This reduces installation size and avoids backend startup issues
on Streamlit Community Cloud.

## Test before deployment

From the project folder:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
python scripts\validate_project.py
python -m pytest -q
python -m streamlit run app\streamlit_app.py
```

## Updating the app

Changes pushed to the relevant project files on the `main` branch normally trigger a Streamlit
rebuild. Dependency-file changes cause package reinstallation.

## Troubleshooting

### ModuleNotFoundError

Confirm that the cloud dependency file exists beside the app:

```text
02-bitcoin-price-prediction/app/requirements.txt
```

### Model artifact not found

Confirm these files are committed:

```text
models/bitcoin_lstm_weights.npz
models/bitcoin_scaler.pkl
models/best_config.json
models/model_metadata.json
```

### Optional live data fails

This is expected when the provider blocks or rate-limits a request. Select the packaged sample. The
application should not fail because of live-data availability.

### Uploaded CSV fails validation

Use at least 60 daily rows and include a date column plus a closing-price column. See
`data/README_data.md`.

## Financial disclaimer

The deployed application must retain the disclaimer. The forecast is an educational machine-learning
demonstration and must not be presented as financial, trading, or investment advice.
