# Project Audit

## Portfolio readiness

- Chronological train, validation, and test handling
- Training-only preprocessing and scaling
- Saved model and preprocessing artifacts
- Baseline forecasting comparison
- Regression and forecasting metrics
- Residual and forecast visualizations
- Pre-trained Streamlit inference app
- Local run scripts for Windows and macOS/Linux
- Docker configuration
- Automated pytest suite
- Dedicated GitHub Actions workflow

## Current held-out results

| Metric | Result |
|---|---:|
| MAE | 13.74 |
| RMSE | 18.70 |
| MAPE | 3.00% |
| R² | 0.937 |

## Validation commands

```bash
python scripts/validate_project.py
python -m compileall -q app src scripts tests train_model.py
python -m pytest -q tests
```
