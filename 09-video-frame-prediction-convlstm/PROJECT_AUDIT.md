# Project Audit

## Audit conclusion

The attached notebook is a valid **single-step next-frame prediction** experiment using a two-layer `ConvLSTM2D` architecture. It is not trained on real videos, Moving MNIST, or surveillance footage. The actual dataset is generated synthetically inside the notebook.

## Verified notebook facts

| Item | Verified value |
|---|---|
| Samples | 2,500 |
| Input frames | 6 |
| Frame size | 32 × 32 |
| Channels | 1 grayscale channel |
| Target | Frame 7, predicted from frames 1–6 |
| Train / validation / test | 1,750 / 375 / 375 |
| Architecture | ConvLSTM2D → BatchNorm → ConvLSTM2D → BatchNorm → Conv2D → Conv2D |
| Parameters | 117,025 |
| Loss | MSE |
| Optimizer | Adam, learning rate 0.001 |
| Epochs | 15 |

## Reproduced test results

| Approach | MAE | RMSE | SSIM | PSNR (dB) | IoU | Pixel accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.023896 | 0.154583 | 0.878204 | 16.3512 | 0.350772 | 0.976104 |
| Frame average | 0.042238 | 0.163222 | 0.710887 | 15.7646 | 0.038070 | 0.964557 |
| ConvLSTM | **0.013383** | **0.044957** | 0.546353 | **31.2984** | **0.919600** | **0.997896** |

## Interpretation

ConvLSTM substantially improves pixel error, PSNR, foreground overlap, and thresholded accuracy. Persistence achieves a higher SSIM because the images are extremely sparse and consecutive square positions overlap enough for the global structural metric to remain high. This reinforces why image metrics must be interpreted together rather than used in isolation.

## Gaps in the original notebook

- Generic dataset description did not identify the actual synthetic generator.
- No reusable preprocessing or inference modules.
- No hosted demo.
- No video or frame ZIP input path.
- No SSIM, PSNR, frame-average baseline, or downloadable outputs.
- No explicit responsible-use statement.
- No deployment, testing, Docker, monorepo, or artifact-management documentation.
