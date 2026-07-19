# Hosting Guide — Streamlit Community Cloud

## Recommended platform

**Streamlit Community Cloud** is the primary recommendation because the project already uses a Streamlit entrypoint, the model is only about 1.4 MB, the safe demo data is small, and deployment can point directly to a subdirectory inside the `lstm-projects` monorepo.

## Required files

- `app/streamlit_app.py`
- `app/requirements.txt`
- `models/convlstm_video_prediction.keras`
- `models/model_metadata.json`
- `data/sample_sequences.npz`
- all imported `src/` modules

## Deployment steps

1. Copy `09-video-frame-prediction-convlstm/` into the root of `lstm-projects/`.
2. From the monorepo root, commit and push the folder to GitHub.
3. Sign in to Streamlit Community Cloud and choose **Create app**.
4. Select the GitHub repository and branch.
5. Enter this entrypoint path:

   ```text
   09-video-frame-prediction-convlstm/app/streamlit_app.py
   ```

6. Open advanced settings and use Python 3.12 for parity with the Dockerfile and local recommendation.
7. Deploy, wait for dependencies and the model to load, and test all three input modes.
8. Replace the placeholder demo URL in the project README and the main monorepo README.

## Important monorepo path rule

Run locally from the repository root when testing the exact cloud path:

```bash
streamlit run 09-video-frame-prediction-convlstm/app/streamlit_app.py
```

The app resolves artifacts from its own project directory, so it does not depend on the current working directory.

## Smoke-test checklist

- Preloaded sample renders six input frames.
- Prediction button returns a next frame.
- Actual/predicted/error comparison appears for samples.
- Multi-step slider produces the selected number of future frames.
- PNG and GIF downloads work.
- The provided `data/sample_frame_sequence.zip` works in ZIP mode.
- A short non-sensitive test video produces a prediction without retraining.

## Troubleshooting

- **Dependency build fails:** confirm the app is using only `app/requirements.txt` and Python 3.12.
- **Model not found:** verify the model file was committed and the entrypoint path is correct.
- **OpenCV error:** keep `opencv-python-headless`, not desktop `opencv-python`, in cloud requirements.
- **Resource limit:** use the safe sample flow and avoid long videos; the app reads only a small number of frames.
- **Theme not applied:** Streamlit Community Cloud recognizes the repository-root `.streamlit/config.toml`; merge this project's config at the monorepo root.

## Alternative: Hugging Face Spaces

A Docker-based Space can use the included `Dockerfile`. Choose the Docker SDK, upload the project, and expose port 8501. Streamlit Community Cloud remains simpler for this GitHub-centered portfolio.
