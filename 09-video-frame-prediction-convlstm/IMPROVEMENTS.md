# Improvements Made to the Original Project

## Original strengths retained

- Reproducible synthetic moving-object generator with seed 42.
- Correct five-dimensional ConvLSTM input shape.
- Independent 70% / 15% / 15% train, validation, and test split.
- Persistence baseline, trained ConvLSTM, residual diagnostics, IoU, and pixel accuracy.
- Lightweight trained `.keras` model suitable for portfolio deployment.

## Portfolio and engineering improvements

1. Reorganized the notebook into modular `src/` components for preprocessing, frame extraction, sequence generation, training, evaluation, inference, and visualization.
2. Added a polished Streamlit app that loads the saved artifact rather than retraining at startup.
3. Added safe sample data, ordered frame ZIP upload, short-video upload, recursive forecasting, PNG download, and GIF download.
4. Added MSE, MAE, RMSE, SSIM, PSNR, foreground IoU, and pixel accuracy.
5. Added persistence and frame-average baseline comparison.
6. Added honest metric interpretation: sparse backgrounds can inflate pixel accuracy and may make SSIM favor persistence even when motion localization is worse.
7. Added responsible-use warnings to both README and app.
8. Added tests, Docker support, local launch scripts, deployment instructions, model metadata, audit documentation, and a file manifest.
9. Preserved the original notebook and specification under `archive/original/` for traceability.

## Important modeling limitation discovered

The attached model is trained only on synthetic 5 × 5 squares on a 32 × 32 black background. The app can preprocess real video, but such input is out-of-distribution and results should be treated as a demonstration of the inference pipeline rather than reliable forecasting.
