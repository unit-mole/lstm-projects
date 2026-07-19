# Monorepo Integration

## Destination

Copy this complete project folder into the root of `lstm-projects` without changing its name:

```text
lstm-projects/
├── .github/
│   └── workflows/
│       └── 07-industrial-equipment-failure-detection-lstm-autoencoder.yml
└── 07-industrial-equipment-failure-detection-lstm-autoencoder/
```

The project folder uses the same sequential numbering and kebab-case naming convention as the existing portfolio projects.

## GitHub Actions workflow

This project has its own root-level workflow file:

```text
.github/workflows/07-industrial-equipment-failure-detection-lstm-autoencoder.yml
```

Do **not** place the `.github` folder inside the project folder. Merge the supplied `.github/workflows` directory into the existing root-level `.github/workflows` directory of `lstm-projects`.

The workflow runs only when this project or its workflow file changes. It performs:

- required-file validation,
- Python source compilation,
- Ruff code-quality checks,
- automated Pytest tests,
- packaged-model artifact validation.

## Correct final repository structure

```text
lstm-projects/
├── .git/
├── .github/
│   └── workflows/
│       ├── 01-airline-passenger-forecasting.yml
│       ├── 02-bitcoin-price-prediction.yml
│       ├── 03-conversational-chatbot-seq2seq-attention.yml
│       ├── 04-ecg-anomaly-detection-lstm-autoencoder-attention.yml
│       ├── 05-fake-news-detection.yml
│       ├── 06-human-activity-recognition-lstm-attention.yml
│       └── 07-industrial-equipment-failure-detection-lstm-autoencoder.yml
├── 01-airline-passenger-forecasting/
├── 02-bitcoin-price-prediction/
├── 03-conversational-chatbot-seq2seq-attention/
├── 04-ecg-anomaly-detection-lstm-autoencoder-attention/
├── 05-fake-news-detection/
├── 06-human-activity-recognition-lstm-attention/
├── 07-industrial-equipment-failure-detection-lstm-autoencoder/
├── .gitignore
├── LICENSE
└── README.md
```

## Main repository README row

```markdown
| 07 | [Industrial Equipment Failure Detection using LSTM Autoencoder](./07-industrial-equipment-failure-detection-lstm-autoencoder) | Predictive maintenance, multivariate anomaly detection, reconstruction-error thresholding | [Live Demo](YOUR_STREAMLIT_URL) |
```

## Suggested main repository description

> Portfolio-grade LSTM projects for forecasting, NLP, anomaly detection, and industrial AI, with reproducible pipelines and interactive Streamlit demos.

## Suggested GitHub topics

```text
lstm, deep-learning, time-series, anomaly-detection, predictive-maintenance,
industrial-iot, sensor-data, streamlit, machine-learning, quality-analytics
```

## Streamlit monorepo note

For Community Cloud, select this entrypoint:

```text
07-industrial-equipment-failure-detection-lstm-autoencoder/app/streamlit_app.py
```

A minimal dependency file is included beside the entrypoint at `app/requirements.txt`. Streamlit configuration is only read from the repository root in a monorepo, so merge this project's `.streamlit/config.toml` settings into the monorepo-level `.streamlit/config.toml` when the same theme is required online.
