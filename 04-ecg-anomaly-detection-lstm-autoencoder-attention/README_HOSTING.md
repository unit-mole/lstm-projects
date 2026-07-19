# Streamlit Community Cloud Hosting Guide

## Recommended platform

Streamlit Community Cloud is recommended because the project is already organized in GitHub, uses
a single Streamlit entry point, and loads a compact NumPy inference artifact without retraining.

## Deployment settings

```text
Repository: unit-mole/lstm-projects
Branch: main
Main file path: 04-ecg-anomaly-detection-lstm-autoencoder-attention/app/streamlit_app.py
Python version: 3.12
```

## Dependency file

The cloud requirements file is placed beside the application entry point:

```text
04-ecg-anomaly-detection-lstm-autoencoder-attention/app/requirements.txt
```

It contains only the packages needed for deployed inference and visualization:

```text
streamlit
numpy
pandas
plotly
```

The native Keras model remains available for reproducibility, but the cloud application loads:

```text
models/lstm_autoencoder_ecg_weights.npz
```

## Deployment steps

1. Copy Project 04 and its workflow into the existing `lstm-projects` repository.
2. Run the project validation and automated tests locally.
3. Commit and push the project to the `main` branch.
4. Open Streamlit Community Cloud.
5. Create a new application from `unit-mole/lstm-projects`.
6. Select the `main` branch.
7. Enter the Project 04 entry-point path.
8. Select Python 3.12 in Advanced settings.
9. Deploy the application.
10. Test packaged signals, both normal and anomaly examples, CSV upload, dataset scoring, downloads,
    performance views, attention qualification, and the healthcare disclaimer.
11. Add the final application URL to the project and root README files.

## Required repository files

```text
04-ecg-anomaly-detection-lstm-autoencoder-attention/
├── app/
├── data/
├── models/
├── outputs/
├── src/
└── README.md
```

Files required at runtime must be committed to GitHub. Do not commit Streamlit secrets, private
medical data, or protected health information.

## Updating the deployed app

Changes pushed to the connected branch trigger an app update. Dependency-file changes trigger a
full environment rebuild.

## Troubleshooting

### Missing package

Confirm that every non-standard imported package appears in `app/requirements.txt`.

### Missing artifact

Confirm these files exist in GitHub with exact capitalization:

```text
models/lstm_autoencoder_ecg_weights.npz
models/model_metadata.json
outputs/model_metrics.json
data/sample_ecg_signals.csv
```

### Incompatible upload

The app requires one row per signal and exactly 140 usable numeric signal values.

### Python version

Use the same Python version locally and in Community Cloud. Changing an already deployed app's
Python version may require redeployment.
