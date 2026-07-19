# Model Card — Industrial LSTM Autoencoder

## Intended use

Portfolio demonstration of multivariate sequence anomaly detection for predictive maintenance and quality analytics.

## Architecture

`Input(20×8) → LSTM(64) → LSTM(32 latent) → RepeatVector(20) → LSTM(32) → LSTM(64) → TimeDistributed Dense(8)`

Total trainable parameters: **64,776**.

## Training and evaluation

- Dataset: deterministic synthetic sensor data, 120 units × 70 cycles.
- Split: unit-level 70% train / 15% validation / 15% test.
- Training subset: healthy training windows only.
- Reconstruction loss: mean squared error.
- Detection score: mean absolute reconstruction error.
- Threshold: mean healthy-training error + 3 standard deviations = **0.330976**.
- Test accuracy: **79.85%**.
- Failure recall: **85.81%**.
- Failure precision: **63.98%**.
- Failure F1: **73.30%**.
- ROC-AUC: **88.67%**.

## Important limitations

This model is not calibrated for a real physical asset. It can mistake equipment-to-equipment baseline differences for anomalies, and reconstruction error does not identify root cause. Operational use would require asset-specific validation, drift monitoring, risk governance, maintenance history, sensor-quality controls, and human review.
