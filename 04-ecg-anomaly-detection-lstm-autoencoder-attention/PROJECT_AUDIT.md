# Project Audit

## Supplied files reviewed

- `ECG_Anomaly_Detection_LSTM_Autoencoders_FULL_ELITE.ipynb`
- `lstm_autoencoder_ecg.keras`
- `ecg_meta.json`

## Verified supplied workflow

The notebook:

- generates 3,000 synthetic ECG-like sequences;
- contains 2,200 normal and 800 anomalous samples;
- creates 140-timestep univariate sequences;
- uses a stratified 70% / 15% / 15% split;
- trains only on 1,540 normal training sequences;
- validates reconstruction on 330 normal validation sequences;
- trains a stacked LSTM Autoencoder;
- calculates mean absolute reconstruction error;
- selects the threshold as training-normal mean plus three standard deviations;
- evaluates on 450 untouched synthetic test sequences.

## Supplied architecture

```text
Input 140 × 1
→ LSTM 64, return sequences
→ LSTM 32 latent vector
→ RepeatVector 140
→ LSTM 32, return sequences
→ LSTM 64, return sequences
→ TimeDistributed Dense 1
```

The model contains **62,529 trainable parameters**.

## Important attention finding

The supplied notebook and trained artifact do **not** contain a trainable attention layer.

To avoid overstating the model:

- the deployed application identifies the pretrained artifact as a stacked LSTM Autoencoder;
- the temporal-focus view is explicitly described as post-hoc explainability derived from
  pointwise reconstruction error;
- `src/model_training.py` provides an optional true trainable temporal-attention architecture for
  future retraining.

## Verified supplied results

| Metric | Result |
|---|---:|
| Validation accuracy | 0.9956 |
| Validation ROC-AUC | 1.0000 |
| Test accuracy | 0.9978 |
| Test anomaly precision | 0.9917 |
| Test anomaly recall | 1.0000 |
| Test anomaly F1 | 0.9959 |
| Test ROC-AUC | 1.0000 |
| Threshold | 0.032153 |

The NumPy inference engine reproduces the supplied training-error statistics, threshold, predictions,
and confusion matrix.

## Scope qualification

The dataset and anomaly modes are synthetic and deliberately separable. The excellent results are
valid for this demonstration distribution but should not be interpreted as clinical diagnostic
performance.
