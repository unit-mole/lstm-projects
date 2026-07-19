# Data Documentation

## Dataset used by the supplied notebook

The uploaded notebook did not load an external corpus during its executed run. It used a deterministic synthetic fallback dataset with:

- **2,500 article-summary pairs**
- Columns: `article` and `summary`
- Seed: `42`
- Split: **1,750 train / 375 validation / 375 test**
- Article length: approximately 48–49 cleaned tokens
- Summary length: approximately 9–10 cleaned tokens

Each article combines an organization, an action, a theme, an expected impact, and implementation context. The reference summary follows the corresponding organization-action-theme-impact pattern.

## Included GitHub-safe files

| File | Purpose |
|---|---|
| `sample_articles.csv` | Eight labeled article-summary pairs used by the Streamlit sample workflow |
| `sample_batch.csv` | Small CSV for testing batch summarization |
| `synthetic_dataset_sample.csv` | Thirty safe examples illustrating the corpus schema |

The complete 2,500-row corpus is reproducible through `src/data_preprocessing.py`; it is not required for app startup.

## Expected custom CSV format

Training data must contain:

```text
article,summary
```

The Streamlit batch interface accepts any CSV with at least one text column. A reference-summary column is optional and enables row-level ROUGE scoring.

## Safety and redistribution

No private customer, employee, quality-case, legal, medical, or copyrighted article data is included. Do not commit confidential documents or user uploads. Store private or licensed corpora outside Git and document their source, license, and permitted use before training.
