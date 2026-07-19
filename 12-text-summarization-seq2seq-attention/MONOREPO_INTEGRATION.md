# Monorepo Integration

## Final location

Extract the monorepo-ready package into the existing repository root so the result is:

```text
lstm-projects/
├── .github/
│   └── workflows/
│       └── 12-text-summarization-seq2seq-attention.yml
└── 12-text-summarization-seq2seq-attention/
```

The workflow belongs under the repository-level `.github/workflows/` directory. Do not place it inside the project folder.

## Main repository README row

Add this project to the completed-projects table:

```markdown
| 12 | [Text Summarization using Seq2Seq with Attention](12-text-summarization-seq2seq-attention/) | NLP · Encoder-Decoder LSTM · Additive Attention · ROUGE | [Live Demo](YOUR_STREAMLIT_URL) |
```

## Recommended repository topics

```text
lstm, seq2seq, attention-mechanism, text-summarization, nlp,
encoder-decoder, keras, jax, streamlit, rouge, machine-learning-portfolio
```

## Streamlit Community Cloud entrypoint

```text
12-text-summarization-seq2seq-attention/app/streamlit_app.py
```

Use Python 3.12 and leave secrets empty.

## Git commands after extraction

```powershell
cd "C:\Users\atripathi\OneDrive - Veralto\Desktop\AI Codes\GIT Projects\lstm-projects"
git add "12-text-summarization-seq2seq-attention" ".github/workflows/12-text-summarization-seq2seq-attention.yml" README.md
git commit -m "Add Seq2Seq attention text summarization project"
git pull --rebase origin main
git push origin main
```

Running `git pull --rebase` before the final push avoids a non-fast-forward rejection when GitHub contains newer commits.
