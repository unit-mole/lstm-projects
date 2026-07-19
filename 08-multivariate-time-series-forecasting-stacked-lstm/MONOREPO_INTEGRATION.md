# Monorepo Integration

Place this entire folder directly inside the existing repository:

```text
lstm-projects/
├── .github/
│   └── workflows/
│       ├── 01-airline-passenger-forecasting.yml
│       ├── 02-bitcoin-price-prediction.yml
│       ├── 03-conversational-chatbot-seq2seq-attention.yml
│       ├── 04-ecg-anomaly-detection-lstm-autoencoder-attention.yml
│       ├── 05-fake-news-detection.yml
│       ├── 06-human-activity-recognition-lstm-attention.yml
│       └── 08-multivariate-time-series-forecasting-stacked-lstm.yml
├── 01-airline-passenger-forecasting/
├── 02-bitcoin-price-prediction/
├── 03-conversational-chatbot-seq2seq-attention/
├── 04-ecg-anomaly-detection-lstm-autoencoder-attention/
├── 05-fake-news-detection/
├── 06-human-activity-recognition-lstm-attention/
├── 07-industrial-equipment-failure-detection-lstm-autoencoder/
└── 08-multivariate-time-series-forecasting-stacked-lstm/
```


## GitHub Actions workflow

The workflow for this project belongs at the **main repository root**, not inside the numbered project folder:

```text
lstm-projects/.github/workflows/08-multivariate-time-series-forecasting-stacked-lstm.yml
```

It runs only when Project 08 or its workflow file changes. The CI job performs Python syntax checks, critical Ruff checks, six automated tests, project-artifact validation, and notebook-format validation. `PYTHONPATH=.` is set so the project-level `src` package is resolved correctly during GitHub Actions testing.

## Recommended GitHub repository description

> Production-style LSTM portfolio featuring forecasting, anomaly detection, NLP and sequence modeling projects with modular Python, model evaluation and Streamlit demos.

## Recommended repository topics

`lstm`, `deep-learning`, `time-series`, `tensorflow`, `keras`, `streamlit`, `machine-learning`, `data-science`, `sequence-modeling`, `portfolio-projects`

## Ready-to-copy root `lstm-projects/README.md`

```markdown
# LSTM Projects Portfolio

A professional collection of Long Short-Term Memory projects covering time-series forecasting, anomaly detection, natural language processing, sequence-to-sequence learning and applied decision-support systems.

I currently work as a **Quality Data Scientist** and am building this repository to demonstrate production-oriented skills for Data Science, Machine Learning, Applied AI, Analytics Engineering and Quality Analytics roles.

## Repository objectives

- Build technically sound LSTM projects with clear business framing.
- Prevent time-series and text-model data leakage.
- Compare deep-learning models against meaningful baselines.
- Save reusable model artifacts and evaluation outputs.
- Provide modular source code, tests and deployment-ready Streamlit applications.
- Present each project in a recruiter-friendly GitHub format.

## Completed projects

| No. | Project | Primary technique | Code | Live demo |
|---:|---|---|---|---|
| 01 | Airline Passenger Forecasting | LSTM forecasting | [Open](./01-airline-passenger-forecasting/) | Add URL |
| 02 | Bitcoin Price Prediction | LSTM forecasting | [Open](./02-bitcoin-price-prediction/) | Add URL |
| 03 | Conversational Chatbot | Seq2Seq with attention | [Open](./03-conversational-chatbot-seq2seq-attention/) | Add URL |
| 04 | ECG Anomaly Detection | LSTM autoencoder with attention | [Open](./04-ecg-anomaly-detection-lstm-autoencoder-attention/) | Add URL |
| 05 | Fake News Detection | LSTM text classification | [Open](./05-fake-news-detection/) | Add URL |
| 06 | Human Activity Recognition | LSTM with attention | [Open](./06-human-activity-recognition-lstm-attention/) | Add URL |
| 07 | Industrial Equipment Failure Detection | LSTM autoencoder | [Open](./07-industrial-equipment-failure-detection-lstm-autoencoder/) | Add URL |
| 08 | Multivariate Time Series Forecasting | Stacked LSTM | [Open](./08-multivariate-time-series-forecasting-stacked-lstm/) | Add URL |

## Project 08 highlight

### Multivariate Time Series Forecasting using Stacked LSTM

Forecasts next-hour energy demand using a 24-hour multivariate sequence of historical load, weather and cyclical calendar features. The project includes chronological evaluation, a naive baseline, residual diagnostics, saved model artifacts and a deployment-ready Streamlit app.

**Skills:** Stacked LSTM, multivariate forecasting, feature engineering, leakage control, regression metrics, recursive inference and Streamlit deployment.

## Planned roadmap

- Neural Machine Translation with Attention
- Stock Market Price Prediction
- Text Summarization using Seq2Seq with Attention
- Traffic Flow Prediction using Stacked LSTM
- Video Frame Prediction using Convolutional LSTM
- Weather Forecasting using ConvLSTM

## Technology stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `TensorFlow` · `Keras` · `Matplotlib` · `Plotly` · `Streamlit` · `Pytest` · `Docker`

## Repository organization

Each numbered project is self-contained and follows the same structure:

```text
project-folder/
├── app/
├── data/
├── images/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
├── tests/
├── README.md
├── README_HOSTING.md
├── requirements.txt
├── Dockerfile
└── train_model.py
```

## Skills demonstrated across the portfolio

- Sequential-data preprocessing and leakage control
- LSTM, Stacked LSTM, attention and autoencoder architectures
- Time-series forecasting and anomaly detection
- NLP classification and sequence generation
- Baseline design and model evaluation
- Residual and error-pattern analysis
- Saved-model inference pipelines
- Interactive Streamlit deployment
- Testing, documentation and GitHub project engineering

## Career positioning

These projects connect my current Quality Data Scientist background with broader ML and Applied AI use cases, including sensor analytics, predictive quality, equipment monitoring, demand forecasting, production planning and operational decision support.
```

## Git commands

```bash
git add 08-multivariate-time-series-forecasting-stacked-lstm
git add .github/workflows/08-multivariate-time-series-forecasting-stacked-lstm.yml
git commit -m "Add multivariate Stacked LSTM project and CI workflow"
git push origin main
```
