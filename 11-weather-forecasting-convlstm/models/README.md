# Model Artifacts

- `convlstm_weather_forecast.keras`: supplied pretrained ConvLSTM model used by the Streamlit application.
- `model_metadata.json`: expanded deployment metadata, architecture summary, metrics, shapes, limitations, and responsible-use note.
- `weather_meta_original.json`: original minimal metadata supplied with the model.

The application loads the model directly and does not retrain during startup.
