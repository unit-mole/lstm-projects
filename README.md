# LSTM Projects

A structured portfolio of end-to-end Long Short-Term Memory projects covering time-series
forecasting, sequence modeling, anomaly detection, attention mechanisms, Seq2Seq systems, natural
language processing, and ConvLSTM applications.

**Portfolio status:** 1 deployed project and 1 additional deployment-ready project  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how Long Short-Term Memory networks and related sequence-modeling
architectures can be applied to practical forecasting, anomaly-detection, natural-language-
processing, and spatiotemporal problems.

Each completed project is developed as an end-to-end case study containing:

- a clearly defined business or analytical problem;
- reproducible data preparation and feature engineering;
- sequence-window generation appropriate to the problem;
- leakage-aware training, validation, and test design;
- task-appropriate baseline comparison and evaluation;
- saved preprocessing and model artifacts;
- modular and reusable inference code;
- an interactive Streamlit demonstration where appropriate;
- automated tests and project-specific GitHub Actions CI;
- local execution and deployment guidance;
- an honest discussion of assumptions, limitations, and future improvements.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied
AI, Data Analytics, Quality Analytics, Business Intelligence, and Analytics Engineering roles.

---

## Completed and Deployment-Ready Projects

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Airline Passenger Forecasting](01-airline-passenger-forecasting/) | Time-series regression and passenger-demand forecasting | [Live Demo](https://lstm-projects-qtuxsozwu2g7kp6lpeuclq.streamlit.app/) |
| 2 | [Bitcoin Price Prediction](02-bitcoin-price-prediction/) | Cryptocurrency time-series forecasting | Streamlit deployment pending |

New project folders and workflow files are added only when each project is developed and validated.

---

## Current Portfolio Coverage

### Airline Passenger Forecasting

The first project demonstrates seasonality-aware monthly passenger-demand forecasting, chronological
validation, training-only scaling, baseline comparison, recursive multi-month prediction, testing,
and Streamlit deployment.

### Bitcoin Price Prediction

The second project demonstrates:

- daily OHLCV cryptocurrency preprocessing;
- 7-day and 30-day moving averages;
- daily return and volatility analysis;
- 30-day multivariate LSTM sequences;
- stacked LSTM inference;
- recursive 1-, 7-, 14-, and 30-day forecasting;
- comparison with naive, moving-average, and linear-trend baselines;
- backend-free NumPy cloud inference;
- optional recent `BTC-USD` retrieval with an offline fallback;
- responsible financial-model communication.

> The Bitcoin project is an educational machine-learning demonstration and is not financial advice.

---

## Planned Project Roadmap

| No. | Project | Primary Modeling Area | Status |
|---:|---|---|---|
| 1 | Airline Passenger Forecasting | Time-series demand forecasting | Deployed |
| 2 | Bitcoin Price Prediction | Financial time-series forecasting | Deployment-ready |
| 3 | Conversational Chatbot using Seq2Seq with Attention | Conversational AI | Planned |
| 4 | ECG Anomaly Detection using LSTM Autoencoder | Healthcare anomaly detection | Planned |
| 5 | Fake News Detection | NLP sequence classification | Planned |
| 6 | Human Activity Recognition using LSTM with Attention | Sensor sequence classification | Planned |
| 7 | Industrial Equipment Failure Detection using LSTM Autoencoder | Predictive maintenance | Planned |
| 8 | Multivariate Time-Series Forecasting using Stacked LSTM | Multivariate forecasting | Planned |
| 9 | Neural Machine Translation with Attention | Seq2Seq translation | Planned |
| 10 | Stock Market Price Prediction | Financial forecasting | Planned |
| 11 | Text Summarization using Seq2Seq with Attention | Abstractive NLP generation | Planned |
| 12 | Traffic Flow Prediction using Stacked LSTM | Transportation forecasting | Planned |
| 13 | Video Frame Prediction using Convolutional LSTM | Spatiotemporal prediction | Planned |
| 14 | Weather Forecasting using ConvLSTM | Spatiotemporal forecasting | Planned |

---

## What the Repository Demonstrates

### End-to-End Machine Learning Delivery

Every completed project is structured to move beyond notebook-only experimentation:

- business-problem definition;
- reproducible data preparation;
- feature and sequence engineering;
- training, validation, and test separation;
- LSTM-based model development;
- baseline comparison and evaluation;
- saved preprocessing and model artifacts;
- reusable forecasting pipelines;
- interactive inference;
- downloadable outputs;
- local execution and cloud deployment.

### Correct Sequential Validation

The repository emphasizes chronological splitting, training-only preprocessing, consistent sequence
construction, validation-based selection, untouched final test evaluation where applicable, and
explicit documentation of leakage risks in supplied legacy artifacts.

### Problem-Appropriate Evaluation

Current projects use MAE, RMSE, MAPE, R², residual analysis, training curves, and transparent
baseline comparisons. Future classification, anomaly-detection, Seq2Seq, and ConvLSTM projects will
add metrics appropriate to their tasks.

### Reliable and Reusable Engineering

Projects use modular source files, saved model metadata, safe input validation, automated tests,
project-specific GitHub Actions workflows, cloud-friendly inference, and GitHub-safe artifact
management.

### Responsible Model Communication

Every project documents intended scope and limitations. Financial, operational, healthcare, and
language-model outputs are not presented as production decisions without additional validation,
governance, monitoring, security, and human oversight.

---

## Repository Convention

The repository is organized as a monorepo. Each completed project is self-contained:

```text
lstm-projects/
├── .github/
│   └── workflows/
│       ├── 01-airline-passenger-forecasting.yml
│       └── 02-bitcoin-price-prediction.yml
├── .streamlit/
│   └── config.toml
├── 01-airline-passenger-forecasting/
├── 02-bitcoin-price-prediction/
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   ├── data/
│   ├── images/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── README.md
│   ├── README_HOSTING.md
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── train_model.py
├── .gitignore
├── LICENSE
└── README.md
```

Each project maintains separate dependencies because forecasting, NLP, anomaly detection, and
spatiotemporal projects may require different libraries and deployment configurations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Passenger-demand forecasting | Airline Passenger Forecasting |
| Cryptocurrency forecasting | Bitcoin Price Prediction |
| Monthly and daily time-series preparation | Projects 01 and 02 |
| Sequence-window generation | LSTM forecasting pipelines |
| Seasonal feature engineering | Airline Passenger Forecasting |
| OHLCV and financial feature engineering | Bitcoin Price Prediction |
| Chronological validation | Both completed projects |
| Leakage prevention and documentation | Training-only pipelines and supplied-artifact audits |
| Recursive forecasting | Multi-month airline and multi-day Bitcoin forecasts |
| Baseline comparison | Naive and task-specific benchmarks |
| Regression evaluation | MAE, RMSE, MAPE, R², and residual analysis |
| Interactive inference | Streamlit applications |
| Testing and CI/CD | pytest and project-specific GitHub Actions workflows |

---

## Core Skills Demonstrated

`Long Short-Term Memory Networks` · `Recurrent Neural Networks` · `Sequence Modeling` ·
`Time-Series Forecasting` · `Demand Forecasting` · `Financial Time-Series Analysis` ·
`OHLCV Data Processing` · `Feature Engineering` · `Moving Averages` · `Return Analysis` ·
`Volatility Analysis` · `Sequence Generation` · `Chronological Validation` ·
`Leakage Prevention` · `Recursive Forecasting` · `Baseline Comparison` ·
`Regression Evaluation` · `Residual Analysis` · `Keras` · `JAX` · `NumPy` ·
`scikit-learn` · `pandas` · `Plotly` · `Streamlit` · `Testing` · `GitHub Actions` ·
`CI/CD` · `Business Translation` · `Responsible Financial Communication`

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
