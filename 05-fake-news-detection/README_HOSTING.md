# Hosting Guide — Streamlit Community Cloud

## Recommended option

Use **Streamlit Community Cloud** for the first public deployment. The application is already built with Streamlit and loads the committed model checkpoint directly, so retraining is not required when the app starts.

## Repository layout required for deployment

For this numbered monorepo structure, keep the files arranged as follows:

```text
lstm-projects/
├── .streamlit/
│   └── config.toml
├── .github/
│   └── workflows/
│       └── 05-fake-news-detection.yml
└── 05-fake-news-detection/
    ├── app/
    │   ├── streamlit_app.py
    │   └── requirements.txt
    ├── data/sample_news.csv
    ├── models/
    │   ├── fake_news_lstm.pt
    │   ├── model_metadata.json
    │   ├── tokenizer_config.json
    │   └── vocabulary.json
    └── src/
```

The full training dataset is not required for deployment.

> **Important:** Do not keep `config.toml` inside `.github/workflows`. That directory is only for GitHub Actions YAML files. The Streamlit configuration must be stored at the repository-level path `.streamlit/config.toml`.

## Push the updated files to GitHub

From the parent `lstm-projects` folder:

```bash
git add .
git commit -m "Add fake news detection LSTM project"
git push
```

## Deploy the application

1. Sign in to Streamlit Community Cloud using GitHub.
2. Create a new app.
3. Select the `lstm-projects` repository and the `main` branch.
4. Use this entrypoint path:

```text
05-fake-news-detection/app/streamlit_app.py
```

5. Open **Advanced settings** and select Python 3.12.
6. Deploy the app.
7. Test manual prediction, sample selection, CSV upload, and CSV download.
8. Replace the placeholder demo URL in the project README and main repository README.

## Why `app/requirements.txt` is included

This repository contains multiple applications. Keeping a deployment-specific `requirements.txt` beside the entrypoint allows Streamlit Community Cloud to install only the dependencies required for this app. The project-level `requirements.txt` is retained for normal local use.

Do not deploy using `requirements-train.txt`; that file contains notebook and model-training packages that the live application does not need.

## Run locally from the repository root

```bash
python -m venv .venv
```

Activate the environment and install the runtime dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r 05-fake-news-detection/requirements.txt

# macOS / Linux
source .venv/bin/activate
pip install -r 05-fake-news-detection/requirements.txt
```

Launch the application from the repository root:

```bash
streamlit run 05-fake-news-detection/app/streamlit_app.py
```

## Model size and Git

The PyTorch checkpoint is small enough to commit directly to GitHub. Git LFS is not required for the current model.

## Troubleshooting

- **Module not found:** confirm the entire `src/` directory is committed.
- **Model file not found:** confirm all files under `05-fake-news-detection/models/` are committed with matching capitalization.
- **Dependency error:** confirm `05-fake-news-detection/app/requirements.txt` exists and use Python 3.12.
- **Theme not applied:** confirm the configuration file is at `.streamlit/config.toml` in the repository root.
- **Slow first prediction:** the app caches the model after the initial load.
- **Memory issue:** keep batch uploads reasonably sized.

## Hugging Face Spaces alternative

A Docker-based Hugging Face Space is also possible, but Streamlit Community Cloud is simpler for this project and provides a direct GitHub-to-demo workflow.
