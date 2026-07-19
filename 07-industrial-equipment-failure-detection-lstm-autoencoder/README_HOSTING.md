# Hosting Guide

## Recommended platform: Streamlit Community Cloud

This is the simplest option for the current GitHub portfolio because the app is already written in Streamlit, the repository is the deployment source, and the included portable inference backend avoids installing TensorFlow during app startup.

Official references:

- Deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- File organization: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- Dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Deploy from the `lstm-projects` monorepo

1. Push the entire project folder to GitHub.
2. Sign in to Streamlit Community Cloud and connect the GitHub account that owns the repository.
3. Choose **Create app** and select the `lstm-projects` repository and branch.
4. Set the entrypoint to:

   ```text
   07-industrial-equipment-failure-detection-lstm-autoencoder/app/streamlit_app.py
   ```

5. In Advanced settings, select Python 3.11 for consistent local/training compatibility.
6. Deploy. The platform will read `app/requirements.txt` beside the entrypoint.
7. Test sample-data loading, equipment selection, charts, and CSV download.
8. Add the resulting `streamlit.app` URL to the project README, main repository README, résumé, LinkedIn, and portfolio site.

## Deploy as a standalone repository

The same folder can be pushed as its own repository. Use `app/streamlit_app.py` as the entrypoint; the root `requirements.txt` and `.streamlit/config.toml` will be detected.

## Hugging Face Spaces alternative

Hugging Face's built-in Streamlit SDK is deprecated; use a **Docker Space** with the included `Dockerfile`. Official reference: https://huggingface.co/docs/hub/spaces-sdks-streamlit

1. Create a new Space with SDK **Docker**.
2. Push the complete project contents to the Space repository.
3. Keep the app listening on port 8501 as configured in the Dockerfile.
4. Review build logs and test the public Space URL.

## Deployment troubleshooting

- **Missing package:** add it to both root `requirements.txt` and `app/requirements.txt` for monorepo deployment.
- **Missing model:** confirm `models/lstm_autoencoder_predictive_maintenance.keras`, `model_metadata.json`, and scaler files are committed.
- **Path error:** run and deploy from repository root; the app resolves project paths from its own file location.
- **Memory issue:** keep the NumPy backend as default. TensorFlow is intentionally excluded from hosting requirements.
