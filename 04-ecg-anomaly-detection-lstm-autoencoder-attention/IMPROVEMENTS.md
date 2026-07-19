# Portfolio Improvements

## 1. Cloud-safe inference

The supplied `.keras` model is preserved for reproducibility. Its exact LSTM and dense weights were
exported to `lstm_autoencoder_ecg_weights.npz`, and the deployed app uses a tested NumPy inference
engine. Streamlit does not need to initialize Keras, TensorFlow, JAX, or PyTorch.

## 2. Robust input preparation

The project now supports:

- packaged synthetic signals;
- wide CSV upload;
- signal-column detection;
- optional labels and record identifiers;
- numeric conversion;
- within-row missing-value interpolation;
- duplicate signal removal;
- fixed sequence-shape validation.

## 3. Reusable anomaly-detection service

`ECGInferenceService` supports:

- one-signal reconstruction and interpretation;
- batch scoring;
- reconstruction-error calculation;
- threshold classification;
- anomaly-score calculation;
- downloadable prediction results.

## 4. Evaluation beyond accuracy

The project reports:

- anomaly precision, recall, and F1;
- ROC-AUC and PR-AUC;
- confusion matrix;
- reconstruction-error distributions;
- threshold sensitivity;
- normal and anomaly examples;
- baseline comparison.

## 5. Honest attention implementation

The supplied artifact has no trainable attention layer. The project therefore separates:

- **deployed explainability:** pointwise reconstruction error converted to normalized temporal focus;
- **optional retraining:** a true trainable temporal-attention pooling layer in
  `src/model_training.py`.

## 6. Healthcare-safe communication

The app and README clearly state that the project is:

- synthetic;
- educational;
- not a medical device;
- not a diagnostic tool;
- not suitable for treatment or clinical decisions.

## 7. Production-oriented repository structure

The project includes modular source code, tests, project validation, GitHub Actions CI, hosting
documentation, a cleaned notebook, saved outputs, data-safety guidance, and recruiter-friendly
portfolio descriptions.
