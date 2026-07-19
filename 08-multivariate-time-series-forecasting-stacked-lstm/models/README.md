# Model Artifacts

- `stacked_lstm_energy.keras`: supplied pre-trained Keras model, verified against the notebook's test metrics.
- `scalers.json`: training-fitted StandardScaler statistics for the eight model features and target.
- `model_metadata.json`: model contract, date ranges, architecture, input/output shape and limitations.

The Streamlit app loads these artifacts directly. It does not retrain the model during startup.
