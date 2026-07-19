# Project Audit

## Supplied material reviewed

- `Abstractive_Text_Summarization_Seq2Seq_Attention_FULL_ELITE.ipynb`
- `seq2seq_summarization.keras`
- `encoder_summarization.keras`
- `decoder_summarization.keras`
- `tokenizer_meta.json`
- Portfolio project specification

The original files are preserved under `archive/original/`.

## Verified current objective

The executed notebook implements **abstractive text summarization** using:

- Encoder embedding and LSTM
- Decoder embedding and LSTM
- Keras `AdditiveAttention`
- Concatenated decoder/context representations
- Time-distributed softmax token prediction
- Teacher forcing during training
- Separate encoder and decoder inference models
- Greedy token-by-token decoding

It is not an extractive summarizer, document retriever, or general-purpose LLM.

## Verified dataset

The executed notebook used its deterministic synthetic fallback:

| Property | Verified value |
|---|---:|
| Records | 2,500 |
| Training | 1,750 |
| Validation | 375 |
| Test | 375 |
| Source vocabulary | 88 including padding |
| Target vocabulary | 57 including padding |
| Maximum source length | 49 |
| Maximum target length | 12 |
| Seed | 42 |

## Verified model artifacts

| Artifact | Parameters | Role |
|---|---:|---|
| `seq2seq_summarization.keras` | 296,505 | Teacher-forced training graph |
| `encoder_summarization.keras` | 142,848 | Encodes source tokens and returns states |
| `decoder_summarization.keras` | 153,657 | Generates one target token per inference step |

All three Keras archives were structurally validated and loaded successfully using Keras 3.13.2 with the JAX CPU backend.

## Tokenizer recovery

The uploaded files contained tokenizer dimensions but not serialized tokenizer objects. Inference would not be reproducible from vocabulary sizes alone. The portfolio refactor therefore reconstructed the source and target tokenizers from the exact deterministic data generator, cleaning logic, split seed, and stable frequency ordering used by the notebook.

The recovered tokenizers were validated against the notebook:

- Source vocabulary: `88`
- Target vocabulary: `57`
- `sostok` ID: `2`
- `eostok` ID: `6`
- First-150 prediction BLEU-like mean: `0.806260`, matching the notebook
- First-150 exact-match ratio: `0.126667`, matching the notebook

The portable JSON tokenizers remove dependence on fragile TensorFlow/Keras pickle objects.

## Verified evaluation

The saved models were re-evaluated on all 375 held-out test records:

| Metric | Result |
|---|---:|
| ROUGE-1 F1 | 0.8189 |
| ROUGE-2 F1 | 0.7972 |
| ROUGE-L F1 | 0.8189 |
| BLEU-like mean | 0.8080 |
| Exact match ratio | 0.1253 |
| Mean generated length | 9.36 words |

The high overlap reflects the narrow deterministic synthetic domain and must not be generalized to real news or document summarization.

## Identified gaps in the original project

1. Tokenizer artifacts required for inference were not exported.
2. No deployment application was included.
3. ROUGE metrics and extractive baselines were missing.
4. Batch CSV inference was missing.
5. Input validation, OOV diagnostics, and long-input warnings were missing.
6. Attention weights were not exposed for inspection.
7. Code remained notebook-centric instead of modular.
8. Tests, GitHub Actions, deployment documentation, and responsible-use framing were missing.

## Audit conclusion

The supplied model is valid for demonstrating Seq2Seq LSTM attention mechanics on a small synthetic corpus. The refactor makes it reproducible, testable, deployable, and transparent without presenting it as a production or general-domain summarization system.
