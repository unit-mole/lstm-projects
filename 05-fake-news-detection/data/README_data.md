# Dataset Notes

## Training dataset

This project is based on the **LIAR** benchmark from the UCSB NLP Group. It contains 12,836 short political statements with six truthfulness labels and metadata.

Hugging Face dataset identifier:

```text
ucsbnlp/liar
```

Original paper:

```text
"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection
William Yang Wang, ACL 2017
```

## Why the full dataset is not included

- The public repository should remain lightweight.
- Dataset redistribution terms should be reviewed separately from the project code license.
- The live Streamlit app does not need training data.
- Full political claims can contain names and context that should not be duplicated unnecessarily.

## Reproduce from Hugging Face

```bash
python train.py --source huggingface
```

The loader resolves the six label names before applying the project's binary mapping.

## Local TSV option

Place the original LIAR files under an ignored folder:

```text
data/raw/train.tsv
data/raw/valid.tsv
data/raw/test.tsv
```

Then run the local command shown in the project README.

## Sample file

`sample_news.csv` contains synthetic, illustrative inputs selected to demonstrate both output classes from the current checkpoint. The sample type describes the model prediction, not verified factual truth.
