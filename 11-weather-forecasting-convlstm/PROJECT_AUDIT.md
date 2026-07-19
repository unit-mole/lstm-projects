# Project Audit

## Supplied artifact identification

- Task: next-frame spatiotemporal weather-map prediction
- Inputs: six historical single-channel `24 × 24` grids
- Output: one predicted `24 × 24 × 1` grid
- Data: deterministic synthetic moving weather-intensity systems
- Architecture: two ConvLSTM2D layers, batch normalization, and two Conv2D layers
- Saved parameters: 117,025

## Recorded held-out results

| Metric | Persistence | ConvLSTM |
|---|---:|---:|
| Test MAE | 0.054178 | 0.027281 |
| Test RMSE | 0.080493 | 0.043898 |

The ConvLSTM reduced MAE by 49.6% and RMSE by 45.5% relative to persistence. Thresholded IoU was 0.6623, and pixel accuracy was 0.9816.

## Leakage and split review

The supplied notebook creates each synthetic sample independently, then applies a fixed seeded sample-level split. No overlapping windows are generated from one continuous global series. This is acceptable for reproducing the educational artifact, but real weather deployments must use chronological splits, training-only normalization, and non-overlapping evaluation periods.

## Deployment review

- The app loads saved artifacts; it does not train at startup.
- `app/requirements.txt` supports Streamlit Community Cloud in a monorepo.
- Paths are resolved relative to the project root rather than the working directory.
- Uploads are shape-validated and non-finite values are repaired.
- Safety-critical claims are explicitly excluded.

## CI review

The root workflow validates syntax, Ruff rules, unit tests, the Keras archive structure, notebook JSON, and required project files. TensorFlow is intentionally not required for lightweight CI artifact checks.
