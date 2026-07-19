# Review of the Original Attached Project

## Original objective

The attached notebook implemented binary classification of LIAR political statements using a Keras LSTM.

## Original recorded results

- Test accuracy: 0.5931
- F1 printed by the notebook: 0.0000
- ROC-AUC: 0.5330
- Confusion matrix: `[[1523, 0], [1045, 0]]`

## Root causes found

1. The notebook combined all official splits before preprocessing and later created a new random split.
2. The tokenizer was fitted on all text before the split.
3. Numeric LIAR labels were mapped using an incorrect assumed order.
4. Labels were encoded alphabetically as `fake = 0`, `real = 1`, so the sigmoid represented real probability.
5. Post-padding was used without an embedding mask; the final LSTM state could be dominated by padding.
6. Threshold selection was not tuned.
7. The saved HDF5 format was legacy.
8. The summary overstated real-world capability despite near-random discrimination.

## Rebuild decision

The public app uses a new PyTorch bidirectional LSTM checkpoint with explicit label semantics, train-only vocabulary fitting, masked sequence pooling, validation-based threshold tuning, and honest baseline comparison. The original artifacts are not used for public inference.
