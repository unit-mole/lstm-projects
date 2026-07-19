# Project Audit

## Supplied implementation

The original notebook is a complete educational Seq2Seq attention workflow:

- 3,500 synthetic rows sampled from 20 fixed dialogue pairs
- source and target text cleaning
- start and end response tokens
- separate source and target tokenizers
- teacher-forcing decoder inputs and targets
- encoder-decoder LSTM architecture
- Keras additive attention
- greedy token-by-token inference
- BLEU-like and exact-match reporting
- saved training, encoder, and decoder models

## Main methodological limitation

All 20 exact dialogue pairs occur in training, validation, and test data because rows were
randomly split after the 20 templates were repeatedly sampled. The reported 1.0 BLEU-like and
exact-match scores therefore measure memorization of known templates.

## Portfolio changes

- Reconstructed and saved the exact tokenizer mappings
- Exported the supplied weights for backend-free NumPy inference
- Added out-of-domain detection and a responsible fallback
- Added a retrieval baseline and comparison
- Added an interactive attention visualization
- Added modular preprocessing, inference, evaluation, and retraining code
- Added tests, validation, CI, hosting documentation, and recruiter-friendly README
- Preserved every supplied artifact under `archive/original-project-files/`
