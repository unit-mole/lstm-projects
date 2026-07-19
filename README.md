# LSTM Projects

A structured portfolio of end-to-end Long Short-Term Memory projects covering time-series
forecasting, financial sequence modeling, conversational AI, anomaly detection, attention
mechanisms, Seq2Seq systems, natural language processing, and ConvLSTM applications.

**Portfolio status:** 2 deployed projects and 1 additional deployment-ready project  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how Long Short-Term Memory networks and related sequence-modeling
architectures can be applied to practical forecasting, conversational AI, anomaly detection,
classification, generation, and spatiotemporal problems.

Each completed project includes:

- a clearly defined business or analytical problem;
- reproducible preprocessing and sequence construction;
- leakage-aware training, validation, and test design;
- saved model and preprocessing artifacts;
- reusable prediction, forecasting, or generation code;
- task-appropriate baseline comparison and evaluation;
- an interactive Streamlit demonstration where appropriate;
- automated tests and project-specific GitHub Actions CI;
- local execution and deployment guidance;
- transparent assumptions, limitations, and responsible-use notes.

The portfolio supports career positioning across Data Science, Machine Learning, Applied AI,
Data Analytics, Quality Analytics, Business Intelligence, and Analytics Engineering.

---

## Completed and Deployment-Ready Projects

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Airline Passenger Forecasting](01-airline-passenger-forecasting/) | Passenger-demand time-series forecasting | [Live Demo](https://lstm-projects-qtuxsozwu2g7kp6lpeuclq.streamlit.app/) |
| 2 | [Bitcoin Price Prediction](02-bitcoin-price-prediction/) | Cryptocurrency time-series forecasting | [Live Demo](https://lstm-projects-k2ocmukxfs83e9ntudpdgr.streamlit.app/) |
| 3 | [Conversational Chatbot using Seq2Seq with Attention](03-conversational-chatbot-seq2seq-attention/) | NLP response generation and conversational AI | Streamlit deployment pending |

New folders and workflows are added only when each project is developed and validated.

---

## Current Portfolio Coverage

### Airline Passenger Forecasting

Seasonality-aware monthly forecasting, chronological validation, baseline comparison, recursive
multi-month prediction, testing, and Streamlit deployment.

### Bitcoin Price Prediction

Daily OHLCV preprocessing, financial feature engineering, stacked-LSTM inference, recursive
multi-day forecasts, volatility analysis, baseline comparison, responsible financial communication,
testing, and Streamlit deployment.

### Conversational Chatbot using Seq2Seq with Attention

Text preprocessing, source and target tokenization, teacher forcing, encoder-decoder LSTMs,
additive attention, greedy generation, token confidence, attention visualization, retrieval
baseline, responsible fallback behavior, testing, and deployment-ready Streamlit chat.

---

## Planned Project Roadmap

| No. | Project | Primary Modeling Area | Status |
|---:|---|---|---|
| 1 | Airline Passenger Forecasting | Time-series demand forecasting | Deployed |
| 2 | Bitcoin Price Prediction | Financial time-series forecasting | Deployed |
| 3 | Conversational Chatbot using Seq2Seq with Attention | Conversational AI | Deployment-ready |
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

- problem definition and scope;
- reproducible data preparation;
- feature, token, and sequence engineering;
- model development and evaluation;
- saved preprocessing and model artifacts;
- reusable forecasting and generation pipelines;
- cloud-safe inference;
- interactive Streamlit applications;
- automated testing and GitHub Actions;
- responsible communication of limitations.

### Correct Validation and Leakage Awareness

The repository emphasizes chronological forecasting splits, training-only preprocessing, duplicate
overlap analysis, unique-pair grouping for dialogue retraining, and honest qualification of supplied
legacy metrics.

### Problem-Appropriate Evaluation

Forecasting projects use MAE, RMSE, MAPE, R², residual analysis, and baseline comparison.
Conversational generation uses token loss, token accuracy, BLEU-like scoring, exact match,
generated examples, attention inspection, and retrieval comparison.

### Responsible Model Communication

Financial predictions are not investment advice. Chatbot responses are not reliable high-stakes or
production-support outputs. Every project documents scope, human oversight, and future validation.

---

## Repository Convention

```text
lstm-projects/
├── .github/
│   └── workflows/
│       ├── 01-airline-passenger-forecasting.yml
│       ├── 02-bitcoin-price-prediction.yml
│       └── 03-conversational-chatbot-seq2seq-attention.yml
├── .streamlit/
│   └── config.toml
├── 01-airline-passenger-forecasting/
├── 02-bitcoin-price-prediction/
├── 03-conversational-chatbot-seq2seq-attention/
│   ├── app/
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
│   └── requirements-dev.txt
├── .gitignore
├── LICENSE
└── README.md
```

Each project is self-contained because forecasting, NLP, anomaly detection, and ConvLSTM systems
require different dependencies, artifacts, and deployment configurations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Monthly demand forecasting | Airline Passenger Forecasting |
| Cryptocurrency forecasting | Bitcoin Price Prediction |
| Conversational response generation | Seq2Seq Attention Chatbot |
| Chronological validation | Projects 01 and 02 |
| Dialogue pair leakage analysis | Project 03 |
| Feature engineering | Seasonal, OHLCV, return, and volatility features |
| Text preprocessing | Cleaning, OOV handling, and sequence padding |
| Encoder-decoder LSTMs | Project 03 |
| Additive attention | Project 03 |
| Recursive forecasting | Projects 01 and 02 |
| Greedy token decoding | Project 03 |
| Regression evaluation | MAE, RMSE, MAPE, R², and residuals |
| Generation evaluation | Loss, token accuracy, BLEU-like, exact match, and examples |
| Baseline comparison | Forecasting and conversational baselines |
| Interactive inference | Three Streamlit applications |
| Testing and CI/CD | pytest and project-specific GitHub Actions |

---

## Core Skills Demonstrated

`Long Short-Term Memory Networks` · `Recurrent Neural Networks` · `Sequence Modeling` ·
`Time-Series Forecasting` · `Financial Forecasting` · `Natural Language Processing` ·
`Seq2Seq` · `Encoder-Decoder Architecture` · `Additive Attention` · `Teacher Forcing` ·
`Text Preprocessing` · `Tokenization` · `Vocabulary Management` · `Sequence Padding` ·
`Greedy Decoding` · `Text Generation` · `Attention Visualization` · `Feature Engineering` ·
`Chronological Validation` · `Leakage Analysis` · `Recursive Forecasting` ·
`Baseline Comparison` · `Regression Evaluation` · `Generation Evaluation` ·
`Keras` · `JAX` · `NumPy` · `scikit-learn` · `pandas` · `Plotly` · `Streamlit` ·
`Testing` · `GitHub Actions` · `CI/CD` · `Responsible AI Communication`

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
