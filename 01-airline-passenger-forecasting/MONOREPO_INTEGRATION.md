# Monorepo Integration

This project is the first independent application inside the `lstm-projects` portfolio repository.

## Repository location

```text
lstm-projects/01-airline-passenger-forecasting/
```

## Dependency policy

The project owns its runtime dependencies through `requirements.txt` and its testing dependencies through `requirements-dev.txt`. Future projects will receive their own dependency files when they are built. This avoids forcing unrelated NLP, computer-vision, or forecasting packages into Project 01.

## Continuous integration

The repository-level workflow is stored at:

```text
.github/workflows/01-airline-passenger-forecasting.yml
```

The workflow runs only when this project or its workflow file changes.

## Deployment entry point

```text
01-airline-passenger-forecasting/app/streamlit_app.py
```
