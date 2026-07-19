# Dataset and Safe Demo Data

## Actual dataset used by the attached notebook

The original notebook does not use external videos. It generates **2,500 synthetic moving-object sequences** in memory. Each sequence contains a 5 × 5 white square moving at a constant velocity on a 32 × 32 black canvas. Motion reflects at image boundaries.

| Property | Value |
|---|---:|
| Input tensor | `(samples, 6, 32, 32, 1)` |
| Target tensor | `(samples, 32, 32, 1)` |
| Color format | Grayscale |
| Pixel range | `[0, 1]` |
| Training sequences | 1,750 |
| Validation sequences | 375 |
| Test sequences | 375 |
| Random seed | 42 |
| Prediction task | Single next frame |

## Included files

- `sample_sequences.npz`: 20 safe test sequences, actual next frames, and predictions from the attached trained model.
- `sample_multistep_sequence.npz`: one six-frame input, six actual future frames, and six recursive model forecasts.
- `sample_video_frames/sequence_000/`: seven PNG frames suitable for demonstrating ordered image input.
- `sample_frame_sequence.zip`: the same frame sequence packaged for the app's ZIP-upload workflow.

## Data safety

The repository deliberately excludes private and copyrighted videos. The bundled data is synthetic and contains no people or personally identifiable information. Do not commit uploaded user videos or any dataset whose license does not allow redistribution.
