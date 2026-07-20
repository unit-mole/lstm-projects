# LSTM Projects

A structured portfolio of twelve completed Long Short-Term Memory projects covering time-series forecasting, financial sequence modeling, conversational AI, text classification, abstractive summarization, anomaly detection, sensor analytics, predictive maintenance, and spatiotemporal forecasting.

**Portfolio status:** 12 completed and deployed projects  
**Repository owner:** [Anmol Tripathi](https://github.com/unit-mole)

---

## Portfolio Objective

This repository demonstrates how Long Short-Term Memory networks and related recurrent architectures can be applied to practical sequential-data problems. Each project is developed as an end-to-end case study containing:

- a clearly defined business, analytical, or applied-AI problem;
- reproducible data preparation, feature engineering, tokenization, signal processing, or spatial preprocessing;
- leakage-aware training, validation, and test design;
- LSTM, Stacked LSTM, Bidirectional LSTM, LSTM Autoencoder, Seq2Seq, Attention, or ConvLSTM model development;
- task-appropriate baseline comparison and evaluation;
- saved preprocessing, tokenizer, metadata, weight, and model artifacts;
- modular and reusable forecasting, classification, reconstruction, generation, or inference code;
- an interactive Streamlit demonstration;
- automated tests and project-specific GitHub Actions CI;
- local execution and deployment guidance;
- an honest discussion of assumptions, limitations, responsible use, and future improvements.

The portfolio is designed to demonstrate skills relevant to Data Science, Machine Learning, Applied AI, Data Analytics, Quality Analytics, Business Intelligence, Analytics Engineering, NLP, Computer Vision, and Industrial Analytics roles.

---

## Completed Projects

| No. | Project | Problem Type | Status |
|---:|---|---|---|
| 1 | [Airline Passenger Forecasting](01-airline-passenger-forecasting/) | Seasonal time-series regression and passenger-demand forecasting | [Live Demo](https://lstm-projects-qtuxsozwu2g7kp6lpeuclq.streamlit.app/) |
| 2 | [Bitcoin Price Prediction](02-bitcoin-price-prediction/) | Multivariate financial time-series forecasting | [Live Demo](https://lstm-projects-k2ocmukxfs83e9ntudpdgr.streamlit.app/) |
| 3 | [Conversational Chatbot using Seq2Seq with Attention](03-conversational-chatbot-seq2seq-attention/) | Conversational AI and neural response generation | [Live Demo](https://lstm-projects-s6ttobrjhi6uyvgwvyygnm.streamlit.app/) |
| 4 | [ECG Anomaly Detection using LSTM Autoencoder with Temporal Attention Analysis](04-ecg-anomaly-detection-lstm-autoencoder-attention/) | Healthcare-style signal reconstruction and anomaly detection | [Live Demo](https://lstm-projects-3k2k8kbwyfws9doojmvfwf.streamlit.app/) |
| 5 | [Fake News Detection](05-fake-news-detection/) | NLP binary sequence classification | [Live Demo](https://lstm-projects-ebn4nfredardyuuzskgnpw.streamlit.app/) |
| 6 | [Human Activity Recognition using LSTM with Attention](06-human-activity-recognition-lstm-attention/) | Multivariate sensor-sequence classification | [Live Demo](https://lstm-projects-tyegesrwm2jemjbldq4fju.streamlit.app/) |
| 7 | [Industrial Equipment Failure Detection using LSTM Autoencoder](07-industrial-equipment-failure-detection-lstm-autoencoder/) | Predictive maintenance and multivariate anomaly detection | [Live Demo](https://lstm-projects-kcgnvnpblpu2fqjhw6tzln.streamlit.app/) |
| 8 | [Multivariate Time-Series Forecasting using Stacked LSTM](08-multivariate-time-series-forecasting-stacked-lstm/) | Multivariate energy-demand forecasting | [Live Demo](https://lstm-projects-me6cghesgakawzytkkrrwp.streamlit.app/) |
| 9 | [Video Frame Prediction using Convolutional LSTM](09-video-frame-prediction-convlstm/) | Spatiotemporal computer-vision forecasting | [Live Demo](https://lstm-projects-efpoyil7h98xqzmxe9r9pt.streamlit.app/) |
| 10 | [Traffic Flow Prediction using Stacked LSTM](10-traffic-flow-prediction-stacked-lstm/) | Transportation time-series forecasting | [Live Demo](https://lstm-projects-gutyrjww4ouvee3rfurrnu.streamlit.app/) |
| 11 | [Weather Forecasting using ConvLSTM](11-weather-forecasting-convlstm/) | Spatiotemporal weather-grid forecasting | [Live Demo](https://lstm-projects-mivsjcuhxgq2szsnou7jdc.streamlit.app/) |
| 12 | [Text Summarization using Seq2Seq with Attention](12-text-summarization-seq2seq-attention/) | Abstractive NLP sequence generation | [Live Demo](https://lstm-projects-8ebpgk2kvotr6yjrgguexw.streamlit.app/) |

---

## What the Portfolio Covers

The twelve projects are intentionally varied so that the repository demonstrates multiple forms of LSTM-based sequence modeling rather than one repeated forecasting workflow.

### Time-Series and Financial Forecasting

- **Airline Passenger Forecasting** predicts future monthly passenger demand using seasonality-aware sequence modeling, cyclical calendar information, and recursive multi-month forecasting.
- **Bitcoin Price Prediction** estimates a future cryptocurrency price path from recent closing-price, moving-average, return, and volume sequences.
- **Multivariate Time-Series Forecasting** predicts next-hour energy demand from historical load, weather variables, and calendar signals using a deep Stacked LSTM.
- **Traffic Flow Prediction** forecasts hourly congestion from recent traffic, speed, occupancy, weather, and time-pattern information.

These projects demonstrate chronological splitting, training-only scaling, lag and calendar feature engineering, sequence-window construction, one-step and recursive multi-step prediction, baseline comparison, residual analysis, and responsible forecasting communication.

### Natural Language Processing and Conversational AI

- **Conversational Chatbot using Seq2Seq with Attention** generates short responses token by token with encoder-decoder LSTMs, teacher forcing, additive attention, and separate inference models.
- **Fake News Detection** classifies short statements as real or fake using text preprocessing, sequence padding, a Bidirectional LSTM classifier, class-aware evaluation, and probability-based outputs.
- **Text Summarization using Seq2Seq with Attention** generates concise abstractive summaries from longer passages using source and target tokenizers, encoder-decoder inference, attention, greedy decoding, and beam-search support.

These projects demonstrate text cleaning, tokenization, vocabulary control, sequence padding, embeddings, teacher forcing, attention alignment, classification thresholding, neural text generation, ROUGE-based evaluation, confidence interpretation, fallback behavior, and comparison with classical or extractive baselines.

### Anomaly Detection and Sensor Analytics

- **ECG Anomaly Detection** learns normal ECG-like signal behavior with an LSTM Autoencoder and identifies unusual patterns through reconstruction error and threshold analysis.
- **Human Activity Recognition** classifies activities from accelerometer and gyroscope sequences using stacked LSTMs and temporal attention.
- **Industrial Equipment Failure Detection** learns normal multivariate equipment behavior and flags unusual sensor windows for predictive-maintenance analysis.

These projects demonstrate normal-only training, reconstruction-based anomaly scoring, threshold selection, multivariate sensor preprocessing, sequence classification, temporal attention, class-probability analysis, anomaly diagnostics, and domain-specific responsible-use boundaries.

### Spatiotemporal Modeling with ConvLSTM

- **Video Frame Prediction** predicts the next grayscale frame from an ordered sequence of historical frames and supports recursive future-frame generation.
- **Weather Forecasting** predicts the next weather-intensity grid from previous spatial observations and supports recursive map forecasting.

These projects demonstrate five-dimensional sequence tensors, spatial and temporal feature learning, ConvLSTM2D architectures, frame-order preservation, image and grid normalization, persistence baselines, structural and spatial evaluation, prediction-error heatmaps, animation generation, and interactive visual analysis.

---

## What the Repository Demonstrates

### End-to-End Machine Learning Delivery

Every project is structured to move beyond notebook-only experimentation. The repository demonstrates:

- business and analytical problem definition;
- reproducible data, text, signal, sensor, image, and grid preparation;
- feature, token, sequence, and spatiotemporal engineering;
- training, validation, and test separation;
- LSTM model development and evaluation;
- saved preprocessing, tokenizer, model, weight, threshold, and metadata artifacts;
- reusable prediction, forecasting, generation, reconstruction, and scoring pipelines;
- manual, sample, and batch inference;
- downloadable forecasts, classifications, anomaly scores, summaries, frames, maps, and animations;
- local execution;
- cloud deployment.

### Sequence Modeling with Correct Validation

Sequential data requires careful validation and preprocessing. The repository emphasizes:

- chronological splitting for forecasting projects;
- training-only feature and target scaling;
- non-overlapping future evaluation periods;
- consistent lookback-window construction during training and inference;
- review-, statement-, dialogue-, signal-, or sample-level splitting where appropriate;
- training-only tokenizer, vocabulary, threshold, and preprocessing decisions;
- normal-only autoencoder training for reconstruction-based anomaly detection;
- preservation of frame order within video and weather sequences;
- validation-based model, threshold, or stopping decisions;
- untouched final test evaluation where applicable;
- explicit documentation of leakage, overlap, synthetic-data, and generalization risks.

### Model Evaluation Based on the Actual Problem

The projects use evaluation metrics that match the task rather than relying on one headline score.

Examples include:

- MAE, RMSE, MAPE, sMAPE, R², residual analysis, and baseline comparison for forecasting;
- volatility, return, replay, and forecast-path diagnostics for financial sequence modeling;
- accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, MCC, and confusion matrices for classification;
- reconstruction-error distributions, anomaly precision and recall, F1, ROC-AUC, PR-AUC, and threshold sensitivity for anomaly detection;
- token loss, token accuracy, confidence, attention alignment, qualitative response review, and retrieval comparison for conversational generation;
- ROUGE, compression ratio, out-of-vocabulary analysis, qualitative review, and extractive-baseline comparison for text summarization;
- pixel error, MAE, RMSE, structural similarity, foreground or spatial-event metrics, and error heatmaps for ConvLSTM forecasting;
- persistence, previous-value, seasonal, moving-average, retrieval, classical NLP, and extractive baselines to determine whether the neural model adds measurable value.

### Reliable and Reusable Engineering

The repository includes practices required for dependable inference:

- preprocessing fitted on training data only;
- consistent feature, token, channel, and sequence order between training and prediction;
- saved scalers, tokenizers, label mappings, thresholds, model metadata, Keras models, PyTorch weights, and portable NumPy weights;
- safe handling of missing values, invalid timestamps, unknown tokens, malformed uploads, non-finite arrays, and incompatible shapes;
- modular source files rather than notebook-only logic;
- pretrained application startup without automatic retraining;
- automated tests for important preprocessing and inference paths;
- project-specific GitHub Actions workflows;
- Streamlit deployment from the main repository branch;
- GitHub-safe data, artifact, and dependency management.

### Business and Analytical Translation

The applications do not stop at raw model outputs. Depending on the project, they provide:

- passenger-demand forecasts;
- cryptocurrency forecast paths and volatility context;
- generated chatbot responses and attention views;
- ECG-like anomaly scores and reconstruction visualizations;
- fake-news probabilities and confidence levels;
- predicted human activities and class distributions;
- industrial equipment anomaly indicators and sensor-level diagnostics;
- energy-demand forecasts;
- predicted video frames and motion-error analysis;
- traffic-congestion forecasts and operational traffic bands;
- predicted weather-intensity maps;
- generated summaries and compression statistics;
- model and baseline comparisons;
- error interpretations;
- batch summaries;
- downloadable scored datasets and generated outputs.

This demonstrates the ability to translate technical model outputs into information that can be understood by analysts, engineers, quality teams, operations teams, managers, and other business stakeholders.

### Responsible Model Communication

Each project documents its intended scope and limitations. The repository avoids presenting educational portfolio models as production-ready financial, medical, misinformation, surveillance, predictive-maintenance, transportation, weather, conversational, or document-intelligence systems without additional validation, governance, monitoring, security controls, domain expertise, and human oversight.

---

## Repository Convention

The repository is organized as a monorepo. Each project generally follows this structure:

```text
lstm-projects/
├── .github/
│   └── workflows/
│       └── project-specific-ci.yml
│
├── project-folder/
│   ├── .streamlit/
│   │   └── config.toml
│   ├── app/
│   │   ├── streamlit_app.py
│   │   └── requirements.txt
│   ├── archive/
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
│   ├── README.md
│   ├── README_HOSTING.md
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── train_model.py
│   └── supporting project files
│
├── .gitignore
├── LICENSE
└── README.md
```

The exact files vary by project, but the standards remain consistent:

- reproducible workflows;
- modular code;
- deployable pretrained inference;
- automated validation;
- clear documentation;
- safe repository practices;
- transparent model assumptions and limitations.

---

## Technical Coverage

| Area | Demonstrated Through |
|---|---|
| Seasonal demand forecasting | Airline passenger forecasting |
| Financial sequence forecasting | Bitcoin price prediction |
| Conversational AI | Seq2Seq chatbot with additive attention |
| Healthcare-style signal analytics | ECG reconstruction and anomaly detection |
| NLP binary classification | Fake news detection |
| Multivariate sensor classification | Human activity recognition |
| Predictive maintenance | Industrial equipment failure detection |
| Multivariate operational forecasting | Energy-demand forecasting with Stacked LSTM |
| Spatiotemporal computer vision | Video frame prediction with ConvLSTM |
| Transportation analytics | Traffic flow prediction |
| Spatiotemporal environmental forecasting | Weather-grid forecasting with ConvLSTM |
| Abstractive text generation | Seq2Seq text summarization with attention |
| Stacked LSTM architectures | Bitcoin, human activity, energy-demand, and traffic projects |
| Bidirectional LSTM classification | Fake news detection |
| LSTM Autoencoders | ECG and industrial equipment anomaly detection |
| Encoder-decoder modeling | Conversational chatbot and text summarization |
| Attention mechanisms | Chatbot, human activity recognition, ECG temporal analysis, and text summarization |
| ConvLSTM2D modeling | Video frame and weather-grid prediction |
| Teacher forcing | Chatbot and text summarization |
| Greedy and beam-search inference | Neural response generation and abstractive summarization |
| Recursive forecasting | Airline, Bitcoin, energy-demand, traffic, video, and weather projects |
| Sequence-window generation | Forecasting, sensor, anomaly-detection, video, and weather projects |
| Chronological validation | Airline, Bitcoin, energy-demand, and traffic forecasting |
| Training-only preprocessing | Scaling, tokenization, vocabulary creation, and threshold decisions |
| Normal-only training | ECG and industrial LSTM Autoencoders |
| Classification thresholding | Fake news and anomaly-detection workflows |
| Baseline forecasting | Seasonal, persistence, previous-value, moving-average, and trend baselines |
| Classical NLP and retrieval baselines | Fake news, chatbot, and text summarization projects |
| Regression evaluation | MAE, RMSE, MAPE, sMAPE, R², residual analysis |
| Classification evaluation | Accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, MCC |
| Anomaly evaluation | Reconstruction error, threshold sensitivity, confusion matrices, ROC-AUC, PR-AUC |
| Generative NLP evaluation | Token metrics, ROUGE, compression, attention, and qualitative analysis |
| Spatial evaluation | Pixel error, structural similarity, event or foreground metrics, and error heatmaps |
| Manual inference | Interactive Streamlit input workflows |
| Batch inference | CSV or array upload, sample scoring, and downloadable outputs |
| Model deployment | Twelve Streamlit Community Cloud applications |
| Testing and CI/CD | pytest, validation scripts, compile checks, and project-specific GitHub Actions |

---

## Core Skills Demonstrated

`Long Short-Term Memory` · `LSTM` · `Stacked LSTM` · `Bidirectional LSTM` · `LSTM Autoencoder` · `Encoder-Decoder` · `Seq2Seq` · `Attention Mechanisms` · `Additive Attention` · `Temporal Attention` · `ConvLSTM` · `ConvLSTM2D` · `Sequence Modeling` · `Time-Series Forecasting` · `Financial Forecasting` · `Demand Forecasting` · `Traffic Forecasting` · `Weather Forecasting` · `Spatiotemporal Forecasting` · `Predictive Maintenance` · `Anomaly Detection` · `Reconstruction Error` · `Human Activity Recognition` · `Sensor Analytics` · `Natural Language Processing` · `Text Classification` · `Conversational AI` · `Abstractive Text Summarization` · `Teacher Forcing` · `Greedy Decoding` · `Beam Search` · `Tokenization` · `Vocabulary Management` · `Sequence Padding` · `Embeddings` · `Feature Engineering` · `Cyclical Features` · `Sequence-Window Generation` · `Recursive Forecasting` · `Chronological Validation` · `Leakage Prevention` · `Threshold Selection` · `Precision–Recall Analysis` · `Baseline Comparison` · `Regression Evaluation` · `Classification Evaluation` · `Spatial Evaluation` · `Residual Analysis` · `Error Analysis` · `Responsible AI Communication` · `Privacy-Aware Deployment` · `TensorFlow` · `Keras` · `JAX` · `PyTorch` · `NumPy` · `pandas` · `scikit-learn` · `OpenCV` · `Plotly` · `Matplotlib` · `Streamlit` · `Testing` · `GitHub Actions` · `CI/CD` · `Business Translation`

---

## Author

**Anmol Tripathi**  
Quality Data Scientist | Data Science | Machine Learning | Applied AI | Analytics
