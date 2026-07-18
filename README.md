# LSTM Projects

A professional portfolio of end-to-end Long Short-Term Memory projects covering time-series forecasting, sequence modeling, anomaly detection, attention mechanisms, Seq2Seq systems, and ConvLSTM applications.

## Current Project

| # | Project | Status |
|---:|---|---|
| 01 | [Airline Passenger Forecasting](./01-airline-passenger-forecasting) | Complete |

The remaining LSTM projects will be added one at a time. Each completed project will receive its own numbered folder, dependencies, tests, documentation, deployment files, and GitHub Actions workflow.

## Repository Structure

```text
lstm-projects/
├── .github/
│   └── workflows/
│       └── 01-airline-passenger-forecasting.yml
├── 01-airline-passenger-forecasting/
│   ├── .streamlit/
│   ├── app/
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

## Dependency Approach

Each project is self-contained. Project 01 has its own `requirements.txt`; future projects will get separate requirement files only when they are created. A single shared root-level requirements file is intentionally not used.

For local development, you may create a virtual environment inside the current project folder:

```powershell
cd 01-airline-passenger-forecasting
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Run Project 01

```powershell
cd 01-airline-passenger-forecasting
run_local.bat
```

Or manually:

```powershell
cd 01-airline-passenger-forecasting
streamlit run app/streamlit_app.py
```
