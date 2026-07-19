# Streamlit Community Cloud Deployment

1. Push this folder to `unit-mole/lstm-projects`.
2. Create a new Streamlit Community Cloud app.
3. Select branch `main`.
4. Use entrypoint `06-human-activity-recognition-lstm-attention/app/streamlit_app.py`.
5. Choose Python 3.12 when available and deploy.

The app loads the pretrained `.keras` model and does not retrain at startup. Dependency installation can take several minutes because TensorFlow is included.
