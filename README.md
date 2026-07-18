# LSTM Projects

A structured portfolio of end-to-end Long Short-Term Memory projects covering time-series forecasting, sequence modeling, anomaly detection, attention mechanisms, Seq2Seq systems, natural language processing, and ConvLSTM applications.

**Portfolio status:** 1 completed and deployed project; additional LSTM projects are planned  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how Long Short-Term Memory networks and related sequence-modeling architectures can be applied to practical forecasting, anomaly-detection, natural-language-processing, and spatiotemporal problems.

Each completed project is developed as an end-to-end case study containing:

- a clearly defined business or analytical problem;
- reproducible data preparation and feature engineering;
- sequence-window generation appropriate to the problem;
- leakage-aware training, validation, and test design;
- LSTM, Stacked LSTM, LSTM Autoencoder, Attention, Seq2Seq, or ConvLSTM model development;
- task-appropriate baseline comparison and evaluation;
- saved preprocessing and model artifacts;
- modular and reusable inference code;
- an interactive Streamlit demonstration where appropriate;
- automated tests and project-specific GitHub Actions CI;
- local execution and deployment guidance;
- an honest discussion of assumptions, limitations, and future improvements.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied AI, Data Analytics, Quality Analytics, Business Intelligence, and Analytics Engineering roles.

---

## Completed Projects

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Airline Passenger Forecasting](01-airline-passenger-forecasting/) | Time-series regression and passenger-demand forecasting | [Live Demo](https://lstm-projects-qtuxsozwu2g7kp6lpeuclq.streamlit.app/) |

The remaining projects will be developed and added one at a time. Each new project will receive its own numbered folder, dependency files, documentation, tests, deployment assets, and GitHub Actions workflow.

---

## Current Project: Airline Passenger Forecasting

The first completed project uses a seasonality-aware LSTM to forecast future monthly airline passenger demand from historical observations.

The project demonstrates:

- chronological training, validation, and test periods;
- training-only feature scaling;
- year-over-year log-growth modeling;
- cyclical month-of-year features;
- LSTM sequence-window generation;
- recursive 6-, 12-, 18-, and 24-month forecasting;
- comparison with naive, seasonal-naive, moving-average, and linear-trend baselines;
- MAE, RMSE, MAPE, R², residual analysis, and training-history evaluation;
- saved model, scaler, metadata, and lightweight cloud-inference artifacts;
- CSV upload and downloadable forecast output;
- automated tests, GitHub Actions CI, and Streamlit deployment.

### Held-out test performance

| Metric | Result |
|---|---:|
| MAE | **13.74** |
| RMSE | **18.70** |
| MAPE | **3.00%** |
| R² | **0.937** |

**Live application:**  
[Open the Airline Passenger Forecasting application](https://lstm-projects-qtuxsozwu2g7kp6lpeuclq.streamlit.app/)

---

## Planned Project Roadmap

| No. | Planned Project | Primary Modeling Area | Status |
|---:|---|---|---|
| 2 | Bitcoin Price Prediction | Financial time-series forecasting | Planned |
| 3 | Conversational Chatbot using Seq2Seq with Attention | Conversational AI and sequence generation | Planned |
| 4 | ECG Anomaly Detection using LSTM Autoencoder | Healthcare anomaly detection | Planned |
| 5 | Fake News Detection | NLP sequence classification | Planned |
| 6 | Human Activity Recognition using LSTM with Attention | Sensor sequence classification | Planned |
| 7 | Industrial Equipment Failure Detection using LSTM Autoencoder | Industrial anomaly detection and predictive maintenance | Planned |
| 8 | Multivariate Time-Series Forecasting using Stacked LSTM | Multivariate forecasting | Planned |
| 9 | Neural Machine Translation with Attention | Seq2Seq translation | Planned |
| 10 | Stock Market Price Prediction | Financial forecasting | Planned |
| 11 | Text Summarization using Seq2Seq with Attention | Abstractive NLP generation | Planned |
| 12 | Traffic Flow Prediction using Stacked LSTM | Transportation demand forecasting | Planned |
| 13 | Video Frame Prediction using Convolutional LSTM | Spatiotemporal sequence prediction | Planned |
| 14 | Weather Forecasting using ConvLSTM | Spatiotemporal weather forecasting | Planned |

The roadmap may evolve as individual projects are reviewed and developed. Only completed and validated projects will be presented as finished portfolio work.

---

## What the Portfolio Will Cover

The planned projects are intentionally varied so that the repository demonstrates more than one type of LSTM and sequential-data problem.

### Time-Series and Financial Forecasting

Projects in this area will include:

- airline passenger-demand forecasting;
- Bitcoin price prediction;
- multivariate time-series forecasting;
- stock-market price prediction;
- traffic-flow prediction;
- weather forecasting.

These projects will demonstrate chronological splitting, training-only scaling, leakage prevention, sequence-window generation, regression evaluation, baseline comparison, residual analysis, and single-step or multi-step forecasting.

### Natural Language Processing and Seq2Seq Systems

Planned NLP projects include:

- conversational chatbot development;
- fake-news detection;
- neural machine translation;
- text summarization.

These projects will demonstrate text cleaning, tokenization, vocabulary management, sequence padding, embeddings, encoder-decoder architecture, attention mechanisms, sequence classification, and autoregressive generation.

### Anomaly Detection and Predictive Maintenance

Planned anomaly-detection projects include:

- ECG anomaly detection using an LSTM Autoencoder;
- industrial-equipment failure detection using an LSTM Autoencoder.

These projects will demonstrate reconstruction-based anomaly scoring, threshold selection, time-series segmentation, class-imbalance considerations, failure-pattern analysis, and practical risk interpretation.

### Sensor and Human-Activity Modeling

The Human Activity Recognition project will demonstrate:

- multichannel sensor-sequence preparation;
- temporal feature learning;
- LSTM-based classification;
- attention-based interpretation;
- confusion-matrix and class-level performance analysis.

### Spatiotemporal Modeling with ConvLSTM

The video-frame and weather-forecasting projects will demonstrate:

- image-sequence preparation;
- spatial and temporal feature learning;
- convolutional recurrent architectures;
- multi-step frame or field prediction;
- spatiotemporal error analysis.

---

## What the Repository Demonstrates

### End-to-End Machine Learning Delivery

Every completed project is structured to move beyond notebook-only experimentation. The repository demonstrates:

- business-problem definition;
- reproducible data preparation;
- feature and sequence engineering;
- training, validation, and test separation;
- LSTM-based model development;
- baseline comparison and evaluation;
- saved preprocessing and model artifacts;
- reusable prediction and forecasting pipelines;
- interactive inference;
- downloadable outputs;
- local execution;
- cloud deployment.

### Sequence Modeling with Correct Validation

Sequential data requires careful validation and preprocessing. The repository emphasizes:

- chronological splitting for forecasting problems;
- entity- or document-level splitting for classification problems;
- training-only scaler, tokenizer, and vocabulary fitting;
- consistent sequence construction during training and inference;
- validation-based model and threshold selection;
- untouched final test evaluation where applicable;
- explicit documentation of leakage risks.

### Model Evaluation Based on the Actual Problem

The projects use evaluation metrics that match the task rather than relying on one headline score.

Examples include:

- MAE, RMSE, MAPE, sMAPE, R², and residual analysis for forecasting;
- precision, recall, F1, specificity, ROC-AUC, PR-AUC, and confusion matrices for classification;
- reconstruction error and anomaly-threshold analysis for autoencoder projects;
- BLEU, ROUGE, sequence accuracy, and qualitative analysis for Seq2Seq systems;
- baseline comparisons to determine whether the neural model adds measurable value;
- training and validation curves to assess convergence and overfitting.

### Reliable and Reusable Engineering

The repository includes practices required for dependable inference:

- preprocessing fitted on training data only;
- consistent feature and sequence order between training and prediction;
- saved scalers, tokenizers, metadata, model configurations, and trained artifacts;
- safe handling of missing values, invalid uploads, and incompatible input formats;
- modular source files rather than notebook-only logic;
- automated tests for important preprocessing and prediction paths;
- project-specific GitHub Actions workflows;
- Streamlit deployment from the main repository branch;
- GitHub-safe data and artifact management.

### Business and Analytical Translation

The applications are designed to move beyond raw model outputs. Depending on the project, they will provide:

- future-demand forecasts;
- financial time-series estimates;
- anomaly scores and risk indicators;
- predicted activity classes;
- fake-news probabilities;
- translated or summarized text;
- chatbot responses;
- forecast trend interpretations;
- model and baseline comparisons;
- error interpretations;
- downloadable prediction or forecast results.

This demonstrates the ability to translate technical model outputs into information that can be understood by analysts, engineers, managers, and other business stakeholders.

### Responsible Model Communication

Each project documents its intended scope and limitations. The repository avoids presenting portfolio models as production-ready forecasting, healthcare, industrial, financial, or language systems without additional validation, governance, monitoring, security controls, and human oversight.

---

## Repository Convention

The repository is organized as a monorepo. Each completed project is self-contained and generally follows this structure:

```text
lstm-projects/
├── .github/
│   └── workflows/
│       └── project-specific-ci.yml
├── .streamlit/
│   └── config.toml
│
├── project-folder/
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   ├── data/
│   │   ├── sample_input.csv
│   │   └── README_data.md
│   ├── images/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── .gitignore
│   ├── Dockerfile
│   ├── README.md
│   ├── README_HOSTING.md
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── supporting project files
│
├── .gitignore
├── LICENSE
└── README.md
```

The exact files may vary by project, but the standards remain consistent:

- reproducible workflows;
- modular code;
- project-specific dependencies;
- deployable inference;
- automated validation;
- clear documentation;
- safe repository practices;
- transparent model limitations.

---

## Current Repository Structure

```text
lstm-projects/
├── .github/
│   └── workflows/
│       └── 01-airline-passenger-forecasting.yml
├── .streamlit/
│   └── config.toml
├── 01-airline-passenger-forecasting/
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   ├── archive/
│   ├── data/
│   ├── images/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   ├── tests/
│   ├── Dockerfile
│   ├── README.md
│   ├── README_HOSTING.md
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── run_local.bat
│   ├── run_local.sh
│   └── train_model.py
├── .gitignore
├── LICENSE
└── README.md
```

Only Project 01 and its workflow are currently included. New project folders and workflow files will be added only when those projects are developed.

---

## Dependency Approach

Each project is self-contained.

Project 01 has its own:

- `requirements.txt` for local development and model work;
- `requirements-dev.txt` for testing and development tools;
- `app/requirements.txt` for Streamlit Community Cloud deployment.

Future projects will receive separate dependency files only when they are created. A single shared root-level requirements file is intentionally not used because the planned projects will require different libraries and deployment configurations.

A local virtual environment can be created inside the active project folder:

```bat
cd 01-airline-passenger-forecasting

py -3.12 -m venv .venv

.venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt -r requirements-dev.txt
```

---

## Run Project 01

Clone the repository and enter the project folder:

```bat
git clone https://github.com/unit-mole/lstm-projects.git

cd lstm-projects\01-airline-passenger-forecasting
```

Activate the environment:

```bat
.venv\Scripts\activate.bat
```

Run the tests:

```bat
python -m pytest -q
```

Launch the Streamlit application:

```bat
python -m streamlit run app\streamlit_app.py
```

The local application will normally open at:

```text
http://localhost:8501
```

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Time-series regression | Airline Passenger Forecasting |
| Passenger-demand forecasting | Recursive 6-, 12-, 18-, and 24-month forecasting |
| Sequence-window generation | Twelve-step LSTM growth sequences |
| Seasonal feature engineering | Year-over-year log growth and cyclical month encoding |
| Chronological validation | Separate training, validation, and held-out test periods |
| Leakage prevention | Training-only scaler fitting and untouched final test evaluation |
| LSTM modeling | Compact Keras LSTM with Huber loss |
| Baseline forecasting | Naive, seasonal-naive, moving-average, and linear-trend models |
| Regression evaluation | MAE, RMSE, MAPE, R², and residual analysis |
| Multi-step forecasting | Recursive future passenger-demand generation |
| Data validation | Date parsing, monthly ordering, duplicate handling, and missing-value preparation |
| Interactive inference | Streamlit sample-data and CSV-upload workflows |
| Downloadable output | Forecast CSV generation |
| Model deployment | Streamlit Community Cloud |
| Testing and CI/CD | pytest and project-specific GitHub Actions workflow |

This table currently reflects the completed Airline Passenger Forecasting project. It will expand as additional LSTM projects are developed.

---

## Core Skills Demonstrated

`Long Short-Term Memory Networks` · `Recurrent Neural Networks` · `Sequence Modeling` ·
`Time-Series Forecasting` · `Demand Forecasting` · `Feature Engineering` ·
`Sequence-Window Generation` · `Trend Analysis` · `Seasonality Analysis` ·
`Chronological Validation` · `Leakage Prevention` · `Training-Only Scaling` ·
`Recursive Forecasting` · `Baseline Comparison` · `Regression Evaluation` ·
`Residual Analysis` · `Error Analysis` · `Keras` · `JAX` · `NumPy` ·
`scikit-learn` · `pandas` · `Plotly` · `Streamlit` · `Model Deployment` ·
`Testing` · `GitHub Actions` · `CI/CD` · `Business Translation`

As the roadmap progresses, the repository will additionally demonstrate LSTM Autoencoders, Stacked LSTM, Attention, Seq2Seq, Natural Language Processing, Human Activity Recognition, Convolutional LSTM, and spatiotemporal modeling.

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
