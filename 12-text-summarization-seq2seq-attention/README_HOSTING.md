# Streamlit Hosting Guide

## Recommended platform

Use **Streamlit Community Cloud** for this project because the application is already written in Streamlit, the model artifacts are small, no API keys are required, and the app loads pretrained encoder/decoder files instead of training during startup.

## Required GitHub locations

```text
lstm-projects/
├── .github/
│   └── workflows/
│       └── 12-text-summarization-seq2seq-attention.yml
└── 12-text-summarization-seq2seq-attention/
    ├── app/
    │   ├── requirements.txt
    │   └── streamlit_app.py
    ├── data/
    │   └── sample_articles.csv
    ├── models/
    │   ├── encoder_summarization.keras
    │   ├── decoder_summarization.keras
    │   ├── source_tokenizer.json
    │   ├── target_tokenizer.json
    │   └── model_metadata.json
    └── src/
```

Do not deploy only the `app/` folder. The app imports modules and loads artifacts from the complete project directory.

## Local cloud-style test

Run from the `lstm-projects` repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r "12-text-summarization-seq2seq-attention\app\requirements.txt"
$env:KERAS_BACKEND="jax"
python -m streamlit run "12-text-summarization-seq2seq-attention\app\streamlit_app.py"
```

Test the preloaded sample first, then manual input, attention expansion, CSV upload, and CSV download.

## Community Cloud deployment values

1. Sign in to Streamlit Community Cloud with the GitHub account that can access `unit-mole/lstm-projects`.
2. Create a new application from an existing repository.
3. Enter:

```text
Repository: unit-mole/lstm-projects
Branch: main
Main file path: 12-text-summarization-seq2seq-attention/app/streamlit_app.py
Python: 3.12
Secrets: none
```

4. Choose a concise app URL such as:

```text
anmol-seq2seq-text-summarizer
```

5. Deploy and watch the build logs.

The app-local dependency file is:

```text
12-text-summarization-seq2seq-attention/app/requirements.txt
```

## Post-deployment checks

Verify these items in the hosted app:

- Application title and responsible-use notice appear.
- A preloaded sample generates a nonempty summary.
- Input, summary, compression, and OOV metrics display.
- Reference ROUGE metrics display for sample data.
- The attention expander renders for greedy decoding.
- Beam search returns a summary.
- `data/sample_batch.csv` uploads successfully.
- Batch results can be downloaded as CSV.
- The GitHub project button opens the Project 12 folder.

## Update the README after deployment

Replace the deployment-pending links in `README.md` with the final URL:

```markdown
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](YOUR_STREAMLIT_URL)

**Live demo:** [Open the Streamlit application](YOUR_STREAMLIT_URL)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_URL)
```

Commit and push the update:

```powershell
git add "12-text-summarization-seq2seq-attention/README.md"
git commit -m "Add live Streamlit demo link for text summarization project"
git pull --rebase origin main
git push origin main
```

## Common problems

### Model file not found

Confirm these files are visible on GitHub:

```text
models/encoder_summarization.keras
models/decoder_summarization.keras
models/source_tokenizer.json
models/target_tokenizer.json
models/model_metadata.json
```

### `ModuleNotFoundError: src`

Use the complete monorepo entrypoint exactly:

```text
12-text-summarization-seq2seq-attention/app/streamlit_app.py
```

The application adds its project root to `sys.path`; deploying a copied or renamed app file can break that path logic.

### Keras backend error

The application sets:

```text
KERAS_BACKEND=jax
```

Confirm `keras==3.13.2` and `jax[cpu]` are present in `app/requirements.txt`. Do not import Keras before the backend environment variable is set.

### Memory or startup delay

Wait for the first model compilation and app startup to finish. The app loads two small inference models and does not load the larger training model. Avoid adding large NLP packages or datasets to the deployment requirements.

### Batch processing is slow

Use greedy decoding and keep uploads below the application's 100-row limit. Beam search runs multiple decoder candidates per token and is intentionally slower.

## Maintenance

- Pushes to `main` will update the deployed app.
- Dependency changes trigger a fresh environment build.
- Keep private data and Streamlit secrets out of Git.
- Retest model loading after changing Keras or JAX versions.
- Preserve the tokenizer JSON files whenever replacing model artifacts.
