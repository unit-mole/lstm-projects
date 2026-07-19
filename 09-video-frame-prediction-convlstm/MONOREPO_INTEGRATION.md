# Monorepo Integration Guide

Place this folder directly inside the existing repository:

```text
lstm-projects/
├── .git/
├── .github/
│   └── workflows/
│       └── 09-video-frame-prediction-convlstm.yml
├── 01-airline-passenger-forecasting/
├── 02-bitcoin-price-prediction/
├── 03-conversational-chatbot-seq2seq-attention/
├── 04-ecg-anomaly-detection-lstm-autoencoder/
├── 05-fake-news-detection/
├── 06-human-activity-recognition-lstm-attention/
├── 07-industrial-equipment-failure-detection-lstm-autoencoder/
├── 08-multivariate-time-series-forecasting-stacked-lstm/
└── 09-video-frame-prediction-convlstm/
```

## Recommended GitHub repository description

> A professional collection of end-to-end LSTM projects covering forecasting, NLP, anomaly detection, computer vision, sequence modeling, evaluation, and Streamlit deployment.

## Recommended GitHub topics

`lstm`, `convlstm`, `deep-learning`, `computer-vision`, `video-prediction`, `spatiotemporal-forecasting`, `keras`, `streamlit`, `machine-learning-portfolio`, `data-science`

## Ready-to-paste main repository README update

```markdown
# LSTM Projects

A professional portfolio of end-to-end Long Short-Term Memory projects spanning time-series forecasting, natural language processing, anomaly detection, computer vision, spatiotemporal modeling, model evaluation, and Streamlit deployment.

I currently work as a **Quality Data Scientist** and am building this repository to demonstrate production-oriented Data Science, Machine Learning, Applied AI, Quality Analytics, and Analytics Engineering skills beyond routine notebook experimentation.

## Completed Projects

| No. | Project | Primary Area | Core Techniques | Live Demo | Status |
|---:|---|---|---|---|---|
| 01 | [Airline Passenger Forecasting](./01-airline-passenger-forecasting/) | Time-Series Forecasting | LSTM, sequence windows, regression metrics | [Demo](ADD_LINK) | Complete |
| 02 | [Bitcoin Price Prediction](./02-bitcoin-price-prediction/) | Financial Forecasting | LSTM, scaling, recursive prediction | [Demo](ADD_LINK) | Complete |
| 03 | [Conversational Chatbot using Seq2Seq with Attention](./03-conversational-chatbot-seq2seq-attention/) | NLP | Encoder-decoder, attention, text generation | [Demo](ADD_LINK) | Complete |
| 04 | [ECG Anomaly Detection using LSTM Autoencoder](./04-ecg-anomaly-detection-lstm-autoencoder/) | Anomaly Detection | LSTM autoencoder, reconstruction error | [Demo](ADD_LINK) | Complete |
| 05 | [Fake News Detection](./05-fake-news-detection/) | NLP Classification | Text preprocessing, LSTM classification | [Demo](ADD_LINK) | Complete |
| 06 | [Human Activity Recognition using LSTM with Attention](./06-human-activity-recognition-lstm-attention/) | Sensor Analytics | Multivariate sequences, attention, classification | [Demo](ADD_LINK) | Complete |
| 07 | [Industrial Equipment Failure Detection using LSTM Autoencoder](./07-industrial-equipment-failure-detection-lstm-autoencoder/) | Predictive Maintenance | Multivariate anomaly detection, reconstruction thresholds | [Demo](ADD_LINK) | Complete |
| 08 | [Multivariate Time-Series Forecasting using Stacked LSTM](./08-multivariate-time-series-forecasting-stacked-lstm/) | Forecasting | Stacked LSTM, multi-feature windows | [Demo](ADD_LINK) | Complete |
| 09 | [Video Frame Prediction using ConvLSTM](./09-video-frame-prediction-convlstm/) | Computer Vision | ConvLSTM2D, next-frame prediction, recursive forecasting | [Demo](ADD_LINK) | Complete |

## Repository Organization

Each project is independently runnable and follows a consistent structure with:

- modular `src/` code,
- a cleaned notebook,
- model artifacts and metadata,
- evaluation outputs,
- automated tests,
- a professional project README,
- local run scripts,
- Docker support,
- Streamlit deployment instructions.

## Technology Stack

Python · NumPy · pandas · scikit-learn · Keras · TensorFlow/JAX · Matplotlib · OpenCV · Streamlit · Docker · pytest

## Skills Demonstrated

- sequence modeling and temporal feature engineering,
- forecasting and anomaly detection,
- encoder-decoder and attention architectures,
- ConvLSTM computer vision and spatiotemporal forecasting,
- baseline design and model evaluation,
- error analysis and responsible-use communication,
- reusable ML pipelines and artifact management,
- interactive deployment and portfolio presentation.

## Roadmap

- Neural Machine Translation with Attention
- Text Summarization using Seq2Seq with Attention
- Traffic Flow Prediction using Stacked LSTM
- Weather Forecasting using ConvLSTM
- Additional modern sequence-model comparisons
```

## Root README project row only

```markdown
| 09 | [Video Frame Prediction using ConvLSTM](./09-video-frame-prediction-convlstm/) | Computer Vision, Spatiotemporal Forecasting | ConvLSTM2D, Keras, Streamlit | [Live Demo](YOUR_STREAMLIT_URL) | Complete |
```

## GitHub Actions workflow

Copy the included workflow file to the monorepo-level workflow directory:

```text
lstm-projects/.github/workflows/09-video-frame-prediction-convlstm.yml
```

The workflow is intentionally stored at the repository root rather than inside the project folder. It runs only when Project 09 or its workflow file changes and performs:

- Python 3.12 environment setup,
- dependency installation from `09-video-frame-prediction-convlstm/requirements-dev.txt`,
- source compilation,
- required-artifact validation,
- the complete pytest suite,
- saved ConvLSTM model loading and inference smoke testing.

Do not place `.github/workflows/` inside `09-video-frame-prediction-convlstm/`; GitHub reads workflows from the root `.github/workflows/` directory.

## Streamlit Community Cloud entrypoint

Use this path when deploying from the monorepo:

```text
09-video-frame-prediction-convlstm/app/streamlit_app.py
```

Community Cloud starts apps from the repository root. The application resolves model and data paths from `__file__`, so paths remain stable locally and in the monorepo. The app-specific dependency file is located beside the entrypoint at `app/requirements.txt`.

Only one `.streamlit/config.toml` is recognized for the repository. Copy or merge this project's theme into the monorepo root `.streamlit/config.toml` before deployment.
