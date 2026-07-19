# Streamlit Community Cloud Hosting

## Recommended platform

Streamlit Community Cloud is the simplest option because the application is already implemented in Streamlit, the pretrained model is committed with the project, and the demo does not need secrets.

## Required GitHub paths

```text
11-weather-forecasting-convlstm/app/streamlit_app.py
11-weather-forecasting-convlstm/app/requirements.txt
11-weather-forecasting-convlstm/models/convlstm_weather_forecast.keras
11-weather-forecasting-convlstm/models/model_metadata.json
11-weather-forecasting-convlstm/data/sample_weather_sequences.npz
11-weather-forecasting-convlstm/src/
```

## Deployment settings

- Repository: `unit-mole/lstm-projects`
- Branch: `main`
- Main file path: `11-weather-forecasting-convlstm/app/streamlit_app.py`
- Python version: `3.11`
- Secrets: none required

## Steps

1. Push the project folder and root workflow to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Select **Create app** and choose `unit-mole/lstm-projects`.
4. Select branch `main`.
5. Enter the exact main file path above.
6. Open Advanced settings and choose Python 3.11.
7. Deploy and monitor the build logs.
8. Confirm that TensorFlow, Keras, scikit-image, and imageio install from `app/requirements.txt`.
9. Test sample selection, next-frame prediction, recursive forecasting, metrics, and downloads.
10. Add the final Streamlit URL to the badges and deployment section in `README.md`.

## Common issues

- `ModuleNotFoundError`: confirm the dependency is listed in `app/requirements.txt`.
- Model not found: confirm the `.keras` file is committed and the entrypoint path is exact.
- Memory limits: keep the sample dataset small and avoid training during app startup.
- Slow first load: TensorFlow model loading may take several seconds on a sleeping Community Cloud instance.
