# Model Artifacts

- `convlstm_video_prediction.keras` — trained Keras model supplied with the original notebook.
- `model_metadata.json` — enriched preprocessing, dataset, architecture, training, and responsible-use metadata.
- `model_metrics.json` — ConvLSTM and baseline results reproduced on the exact 375-sequence test split.

The model expects a float32 tensor with shape `(batch, 6, 32, 32, 1)` and returns `(batch, 32, 32, 1)`.
