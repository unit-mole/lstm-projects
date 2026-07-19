# Project Audit

## What the uploaded project actually does

The notebook generates a deterministic synthetic industrial dataset with **120 equipment units, 70 cycles per unit, eight sensor features, and a simulated pre-failure label**. It uses a leakage-resistant unit-level split, fits the scaler on training data only, creates 20-step windows, trains the LSTM Autoencoder on healthy windows, and flags anomalies from mean absolute reconstruction error.

## Strong elements retained

- Equipment-level train/validation/test separation.
- Training-only scaler fitting.
- Normal-only autoencoder training.
- Clear 64→32 latent→32→64 LSTM encoder/decoder architecture.
- Reconstruction-error thresholding and labeled test evaluation.
- Saved Keras model artifact and deterministic seed.

## Gaps found in the original version

1. The work was contained in one long notebook instead of reusable modules.
2. No Streamlit application or inference pipeline was included.
3. The scaler was not saved, so the model could not reproduce training-time preprocessing outside the notebook.
4. Metadata contained only sequence length, feature count, threshold, sensor names, and seed.
5. Plots and metrics were displayed but not persisted as portfolio artifacts.
6. No baseline comparison, PR-AUC, tests, Docker setup, or hosting guide was included.
7. The project language sometimes implied production readiness despite using synthetic data.
8. The global threshold is sensitive to unit-to-unit operating differences. On the test set, 143 of 622 healthy windows were flagged, showing the cost of false positives.

## Verified original-artifact results

| Metric | Value |
|---|---:|
| Validation accuracy | 84.10% |
| Validation ROC-AUC | 93.28% |
| Test accuracy | 79.85% |
| Test ROC-AUC | 88.67% |
| Failure precision | 63.98% |
| Failure recall | 85.81% |
| Failure F1 | 73.30% |
| Threshold | 0.330976 |

## Improvements implemented

- Modular source package and CLI training entrypoint.
- Saved scaler in both pickle and portable JSON formats.
- Expanded metadata and model card.
- Lightweight NumPy inference backend that reads the actual Keras weights, allowing the hosted demo to run without TensorFlow.
- Streamlit upload/sample workflow, equipment selector, sequence inspection, trends, reconstruction comparison, health timeline, sensor contribution, and CSV download.
- Persisted evaluation plots, baseline table, metrics, and predictions.
- Unit tests, Dockerfile, local-run scripts, data-safety guide, and monorepo integration guide.
- Explicit industrial-safety disclaimer and conservative capability wording.
