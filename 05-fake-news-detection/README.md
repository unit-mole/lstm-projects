# Fake News Detection using Bidirectional LSTM

[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)](https://streamlit.io/)

> **Responsible-use notice:** This project is for educational and portfolio demonstration purposes only. It must not be used as the sole source for deciding whether a claim, headline, or article is true or false. The model may be wrong and may reproduce bias from its training data. Real-world verification requires source checking, evidence review, context, and qualified human judgment. Do not use this demo for legal, political, journalistic, financial, medical, public-safety, or other high-impact decisions.

## Project overview

This project estimates whether a **short political claim** is more similar to the LIAR dataset's fake or real training categories. It is a content-pattern classifier, not a fact-checking engine.

**Question:** Given a short claim or headline, can a sequence model estimate whether its language resembles claims labeled fake or real in the training data?

The application returns:

- predicted label,
- fake-news probability,
- confidence score,
- an interpretation statement,
- token-coverage information,
- a responsible-use reminder.

## What the attached project was actually doing

The original notebook loaded the **LIAR dataset**, combined its train/validation/test splits, selected the `statement` and `label` columns, mapped six truthfulness labels into two classes, tokenized the text, trained a single LSTM, and saved a Keras HDF5 model plus pickle artifacts.

The original checkpoint should not be presented as a successful model because its recorded test behavior was:

| Metric | Original notebook |
|---|---:|
| Accuracy | 59.31% |
| Fake/positive-class F1 as coded | 0.000 |
| ROC-AUC | 0.533 |
| Prediction pattern | Every test item assigned to one class |

## Critical issues corrected

1. **Incorrect numeric label mapping.** The original code assumed a numeric class order that did not match the LIAR label names. The rebuilt pipeline converts labels to names first and then uses an explicit mapping.
2. **Data leakage.** The tokenizer was fitted before the train/test split. The rebuilt vocabulary is fitted on the training split only.
3. **Padding/LSTM mismatch.** Post-padding was used without masking, so the final recurrent state could be dominated by padding. The rebuilt model uses masked mean/max pooling over valid sequence positions.
4. **Probability ambiguity.** The original sigmoid output represented the encoded class rather than an explicitly documented fake probability. The rebuilt model defines `fake = 1` throughout.
5. **Weak evaluation.** The rebuilt version adds precision, recall, F1, macro-F1, balanced accuracy, ROC-AUC, PR-AUC, threshold tuning, confusion matrix, and error analysis.
6. **No baseline.** A TF-IDF + logistic-regression baseline is included.
7. **Unsafe artifact dependency.** The deployment path uses JSON metadata and a PyTorch checkpoint instead of requiring pickle files.

## Dataset

The project uses **LIAR**, a benchmark of approximately 12.8K short statements labeled by PolitiFact editors across six truthfulness levels.

### Binary mapping used here

| Original LIAR label | Portfolio class | Numeric target |
|---|---|---:|
| `pants-fire` | Fake | 1 |
| `false` | Fake | 1 |
| `barely-true` | Fake | 1 |
| `half-true` | Real | 0 |
| `mostly-true` | Real | 0 |
| `true` | Real | 0 |

This mapping is a modeling choice, not a universal definition of misinformation. `half-true` is especially ambiguous and contributes label noise.

The full dataset is intentionally excluded from the public project folder. See [`data/README_data.md`](./data/README_data.md).

## Workflow

```text
LIAR train / validation / test splits
                ↓
Explicit string-label mapping and duplicate checks
                ↓
Leakage-safe vocabulary built on training text only
                ↓
Token sequence + fixed-length padding + OOV handling
                ↓
Embedding → Bidirectional LSTM → masked mean/max pooling
                ↓
Dense binary classifier producing P(fake)
                ↓
Validation-based threshold tuning
                ↓
Test evaluation, baseline comparison, error analysis
                ↓
Saved checkpoint + metadata → Streamlit inference app
```

## Text preprocessing

The pipeline intentionally avoids aggressive cleaning. It:

- extracts words, contractions, numbers, URLs, `%`, `$`, `!`, and `?`,
- lowercases lexical tokens,
- replaces URLs and numbers with stable placeholder tokens,
- preserves headline-style punctuation signals,
- adds an all-caps style token when applicable,
- handles unseen words through `<OOV>`,
- pads or truncates to the configured maximum sequence length,
- fits the vocabulary on training data only.

## LSTM architecture

```text
Token IDs
  ↓
Embedding (64 dimensions)
  ↓
Bidirectional LSTM (40 units per direction)
  ↓
Masked mean pooling + masked max pooling
  ↓
Dense(64) + ReLU + Dropout
  ↓
Sigmoid-compatible logit for P(fake)
```

The trained checkpoint contains 6,638 vocabulary entries and uses a maximum sequence length of 48 tokens.

## Class imbalance and threshold selection

The training loss uses a positive-class weight calculated from the training split. The decision threshold was selected on the validation split to maximize **macro-F1**, which gives both classes equal importance. The selected threshold is **0.52**.

## Results

### Test-set comparison

| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1 (Fake) | Macro-F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 63.25% | 58.02% | 54.87% | 56.40% | 62.32% | 67.20% | 60.49% |
| Bidirectional LSTM | 62.24% | 56.92% | 52.71% | 54.73% | 61.17% | 65.30% | 59.35% |

### LSTM confusion matrix

|  | Predicted Real | Predicted Fake |
|---|---:|---:|
| Actual Real | 504 | 221 |
| Actual Fake | 262 | 292 |

The simpler TF-IDF baseline slightly outperformed the LSTM. This is an important and honest result: the LIAR statements are short, the dataset is relatively small, and n-gram signals can be highly competitive. The LSTM remains valuable as a sequence-modeling portfolio implementation, but it is not presented as the superior production model.

## Error analysis

Common failure modes include:

- truthful claims written in sensational or adversarial language,
- false claims written in neutral language,
- claims that cannot be judged without external evidence,
- ambiguous `half-true` examples,
- topic and speaker bias in political data,
- short statements with limited linguistic context,
- out-of-domain news articles that differ from LIAR's political claims.

See [`outputs/error_analysis.csv`](./outputs/error_analysis.csv) and [`outputs/sample_predictions.csv`](./outputs/sample_predictions.csv).

## Streamlit demo

**Live demo:** `ADD_YOUR_STREAMLIT_APP_URL_HERE`

The app supports:

- manual claim/headline entry,
- illustrative sample inputs,
- CSV batch upload,
- automatic text-column detection with manual override,
- downloadable scored CSV,
- prediction-distribution chart,
- model details and limitations.

## Run locally

```bash
cd 05-fake-news-detection
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the deployment dependencies and launch the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

To retrain and reproduce the analysis:

```bash
pip install -r requirements-train.txt
python train.py --source huggingface
```

Local TSV files can also be supplied:

```bash
python train.py \
  --source local \
  --train-path data/raw/train.tsv \
  --validation-path data/raw/valid.tsv \
  --test-path data/raw/test.tsv
```

## Project structure

```text
05-fake-news-detection/
├── app/streamlit_app.py
├── data/
│   ├── README_data.md
│   └── sample_news.csv
├── images/README.md
├── legacy_original/
│   ├── ORIGINAL_PROJECT_REVIEW.md
│   └── original_notebook.ipynb
├── models/
│   ├── fake_news_lstm.pt
│   ├── model_metadata.json
│   ├── tokenizer_config.json
│   └── vocabulary.json
├── notebooks/fake_news_detection.ipynb
├── outputs/
├── src/
├── tests/
├── .streamlit/config.toml
├── .gitignore
├── README.md
├── README_HOSTING.md
├── requirements.txt
├── requirements-train.txt
└── train.py
```

## Deployment

Streamlit Community Cloud is the recommended first option because the app is Streamlit-native, the model is small enough to store in GitHub, and deployment can be connected directly to the repository. Full steps are in [`README_HOSTING.md`](./README_HOSTING.md).

## Limitations

- This is not a factual-verification system and does not retrieve evidence.
- The LIAR domain is U.S. political claims, not general news articles.
- Labels are editorial judgments mapped from six levels into two classes.
- The dataset is small for deep learning.
- Language, source, time period, and topic shifts can reduce performance.
- Probability is model confidence under the training distribution, not probability that a claim is objectively false.
- High-confidence errors remain possible.

## Future improvements

- preserve the original six-class task,
- add calibrated uncertainty and an abstain option,
- evaluate source-disjoint and time-based splits,
- compare GRU, CNN-LSTM, and transformer baselines,
- incorporate evidence retrieval and source credibility as separate modules,
- add SHAP or token-occlusion analysis carefully,
- test on out-of-domain and multilingual datasets,
- add model cards and automated CI tests.

## Skills demonstrated

NLP preprocessing, leakage prevention, vocabulary management, bidirectional LSTM modeling, masked sequence pooling, class weighting, threshold tuning, baseline comparison, evaluation, error analysis, model-risk communication, modular Python, GitHub project design, and Streamlit deployment.
