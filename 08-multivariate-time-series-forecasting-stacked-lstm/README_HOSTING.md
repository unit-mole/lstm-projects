# Hosting Guide

## Recommended option: Streamlit Community Cloud

The project is already structured around Streamlit, uses a pre-trained model and exposes a single entry point: `app/streamlit_app.py`.

### Before deployment

1. Push this folder into your public `lstm-projects` GitHub repository.
2. Confirm that `models/stacked_lstm_energy.keras` is committed and remains below GitHub's normal file-size limit.
3. Confirm that `requirements.txt` is at the project-folder root.
4. Update the GitHub-link placeholder in the app and README.

### Deploy

1. Sign in to Streamlit Community Cloud with GitHub.
2. Select **Create app**.
3. Choose your `lstm-projects` repository and branch.
4. Set the main file path to:

```text
08-multivariate-time-series-forecasting-stacked-lstm/app/streamlit_app.py
```

5. In advanced settings, select Python 3.11 when available.
6. Deploy and review the build logs.
7. Test the sample-data flow, CSV upload, metrics tabs and forecast download.

The resulting public URL can be added to GitHub, LinkedIn, your résumé and portfolio website.

## Troubleshooting

- **TensorFlow installation is slow:** first deployment can take several minutes because the runtime downloads the CPU package.
- **Model loading error:** verify `keras==3.13.2` is installed and the model file is not corrupted by Git LFS pointer text.
- **Out-of-memory build:** use the Dockerfile on a container host or move to Hugging Face Spaces with a Docker/Streamlit setup.
- **App cannot import `src`:** keep the provided folder structure; the app adds the project root to `sys.path`.

## Hugging Face Spaces alternative

Create a new Space using the Docker SDK, upload this project and use the included Dockerfile. This is useful when you want explicit control over Python and TensorFlow versions.
