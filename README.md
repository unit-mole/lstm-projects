# LSTM Projects

A structured portfolio of end-to-end Long Short-Term Memory projects covering time-series
forecasting, financial sequence modeling, conversational AI, healthcare-style anomaly detection,
attention mechanisms, Seq2Seq systems, natural language processing, and ConvLSTM applications.

**Portfolio status:** 3 deployed projects and 1 additional deployment-ready project  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how LSTM networks and related sequence architectures can be applied to
practical forecasting, conversational AI, anomaly detection, classification, generation, and
spatiotemporal problems.

Each completed project includes reproducible preprocessing, sequence construction, model
development, problem-appropriate evaluation, saved artifacts, reusable inference, an interactive
demo, automated tests, CI/CD, deployment guidance, and transparent limitations.

The portfolio supports career positioning across Data Science, Machine Learning, Applied AI,
Data Analytics, Quality Analytics, Business Intelligence, and Analytics Engineering.

---

## Completed and Deployment-Ready Projects

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Airline Passenger Forecasting](01-airline-passenger-forecasting/) | Passenger-demand time-series forecasting | [Live Demo](https://lstm-projects-qtuxsozwu2g7kp6lpeuclq.streamlit.app/) |
| 2 | [Bitcoin Price Prediction](02-bitcoin-price-prediction/) | Cryptocurrency time-series forecasting | [Live Demo](https://lstm-projects-k2ocmukxfs83e9ntudpdgr.streamlit.app/) |
| 3 | [Conversational Chatbot using Seq2Seq with Attention](03-conversational-chatbot-seq2seq-attention/) | NLP response generation and conversational AI | [Live Demo](https://lstm-projects-s6ttobrjhi6uyvgwvyygnm.streamlit.app/) |
| 4 | [ECG Anomaly Detection using LSTM Autoencoder](04-ecg-anomaly-detection-lstm-autoencoder-attention/) | Healthcare-style signal reconstruction and anomaly detection | Streamlit deployment pending |

---

## Current Portfolio Coverage

### Airline Passenger Forecasting

Seasonality-aware monthly forecasting, chronological validation, baseline comparison, recursive
multi-month prediction, testing, and Streamlit deployment.

### Bitcoin Price Prediction

Daily OHLCV preprocessing, financial feature engineering, stacked-LSTM inference, recursive
multi-day forecasting, volatility analysis, baseline comparison, and responsible financial
communication.

### Conversational Chatbot using Seq2Seq with Attention

Text preprocessing, teacher forcing, encoder-decoder LSTMs, additive attention, greedy decoding,
token confidence, attention visualization, retrieval comparison, responsible fallback behavior,
and Streamlit chat deployment.

### ECG Anomaly Detection using LSTM Autoencoder

Normal-only training, synthetic ECG-like signal reconstruction, reconstruction-error thresholding,
anomaly precision and recall, baseline comparison, temporal focus explainability, cloud-safe NumPy
inference, healthcare disclaimers, and deployment-ready Streamlit analysis.

---

## Planned Project Roadmap

| No. | Project | Primary Modeling Area | Status |
|---:|---|---|---|
| 1 | Airline Passenger Forecasting | Time-series demand forecasting | Deployed |
| 2 | Bitcoin Price Prediction | Financial time-series forecasting | Deployed |
| 3 | Conversational Chatbot using Seq2Seq with Attention | Conversational AI | Deployed |
| 4 | ECG Anomaly Detection using LSTM Autoencoder | Healthcare anomaly detection | Deployment-ready |
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

- clear problem framing and responsible scope;
- reproducible data and signal preparation;
- feature, token, and sequence engineering;
- LSTM forecasting, generation, and reconstruction;
- thresholding and baseline comparison;
- saved artifacts and cloud-safe inference;
- interactive Streamlit applications;
- downloadable outputs;
- automated tests and GitHub Actions;
- honest communication of model limitations.

### Validation and Leakage Awareness

The repository emphasizes chronological forecasting splits, training-only preprocessing, explicit
dialogue overlap analysis, normal-only autoencoder training, untouched test evaluation, and
qualification of supplied legacy metrics.

### Problem-Appropriate Evaluation

Current projects use regression metrics and residual analysis for forecasting, token loss and
qualitative examples for conversational generation, and precision, recall, F1, ROC-AUC, PR-AUC,
confusion matrices, reconstruction errors, and threshold sensitivity for anomaly detection.

### Responsible AI Communication

Financial outputs are not investment advice. Chatbot outputs are not reliable high-stakes support.
ECG anomaly outputs are not diagnoses. Every project documents required human oversight, data
governance, external validation, and future monitoring.

---

## Repository Convention

```text
lstm-projects/
├── .github/
│   └── workflows/
│       ├── 01-airline-passenger-forecasting.yml
│       ├── 02-bitcoin-price-prediction.yml
│       ├── 03-conversational-chatbot-seq2seq-attention.yml
│       └── 04-ecg-anomaly-detection-lstm-autoencoder-attention.yml
├── .streamlit/
│   └── config.toml
├── 01-airline-passenger-forecasting/
├── 02-bitcoin-price-prediction/
├── 03-conversational-chatbot-seq2seq-attention/
├── 04-ecg-anomaly-detection-lstm-autoencoder-attention/
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

Each project is self-contained because forecasting, NLP, anomaly detection, and spatiotemporal
systems require different dependencies, artifacts, and deployment configurations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Monthly demand forecasting | Airline Passenger Forecasting |
| Cryptocurrency forecasting | Bitcoin Price Prediction |
| Conversational response generation | Seq2Seq Attention Chatbot |
| Healthcare-style anomaly detection | ECG LSTM Autoencoder |
| Chronological validation | Projects 01 and 02 |
| Dialogue overlap analysis | Project 03 |
| Normal-only model training | Project 04 |
| Feature and signal engineering | Projects 01, 02, and 04 |
| Text preprocessing and tokenization | Project 03 |
| Encoder-decoder LSTMs | Project 03 |
| Reconstruction-error thresholding | Project 04 |
| Attention and temporal focus | Projects 03 and 04 |
| Regression evaluation | Projects 01 and 02 |
| Generation evaluation | Project 03 |
| Anomaly precision-recall evaluation | Project 04 |
| Interactive inference | Four Streamlit applications |
| Testing and CI/CD | pytest and project-specific GitHub Actions |

---

## Core Skills Demonstrated

`Long Short-Term Memory Networks` · `Recurrent Neural Networks` · `Sequence Modeling` ·
`Time-Series Forecasting` · `Financial Forecasting` · `Natural Language Processing` ·
`Seq2Seq` · `Encoder-Decoder Architecture` · `Additive Attention` · `Teacher Forcing` ·
`LSTM Autoencoder` · `Unsupervised Anomaly Detection` · `Signal Reconstruction` ·
`Reconstruction Error` · `Threshold Selection` · `Precision-Recall Analysis` ·
`ROC-AUC` · `PR-AUC` · `Temporal Focus Explainability` · `Feature Engineering` ·
`Chronological Validation` · `Leakage Analysis` · `Recursive Forecasting` ·
`Baseline Comparison` · `Keras` · `JAX` · `NumPy` · `scikit-learn` · `pandas` ·
`Plotly` · `Streamlit` · `Testing` · `GitHub Actions` · `CI/CD` ·
`Responsible AI Communication`

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
