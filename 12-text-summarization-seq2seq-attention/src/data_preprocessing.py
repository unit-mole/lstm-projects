"""Dataset generation, loading, cleaning, and leakage-aware splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import END_TOKEN, SEED, START_TOKEN
from .text_preprocessing import clean_text

SUBJECTS = ["company", "government", "research team", "hospital", "startup", "city council", "university", "energy provider"]
ACTIONS = ["announced", "reported", "launched", "confirmed", "investigated", "expanded", "approved", "published"]
THEMES = ["new policy", "product release", "clinical study", "funding round", "service disruption", "sustainability plan", "safety update", "technology roadmap"]
IMPACTS = ["improving operations", "reducing costs", "supporting citizens", "expanding access", "strengthening compliance", "improving outcomes", "accelerating deployment", "boosting efficiency"]


def generate_summarization_dataset(n: int = 2500, seed: int = SEED) -> pd.DataFrame:
    """Reproduce the deterministic synthetic corpus from the uploaded notebook."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for row_id in range(n):
        subject = str(rng.choice(SUBJECTS))
        action = str(rng.choice(ACTIONS))
        theme = str(rng.choice(THEMES))
        impact = str(rng.choice(IMPACTS))
        article = " ".join(
            [
                f"The {subject} {action} a {theme} after several months of internal review and consultation with stakeholders.",
                f"Officials said the decision is expected to help with {impact} over the next few quarters.",
                "Additional commentary noted that implementation will happen in phases and that teams will monitor risks closely.",
            ]
        )
        summary = f"{subject} {action} a {theme} to support {impact}."
        rows.append({"row_id": row_id, "article": article, "summary": summary})
    return pd.DataFrame(rows)


def load_article_summary_csv(
    path: str | Path,
    article_column: str = "article",
    summary_column: str = "summary",
) -> pd.DataFrame:
    """Load and validate an article-summary CSV without silently guessing columns."""

    frame = pd.read_csv(path)
    missing = {article_column, summary_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame[[article_column, summary_column]].rename(
        columns={article_column: "article", summary_column: "summary"}
    )


def prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean, de-duplicate, and add decoder boundary tokens."""

    required = {"article", "summary"}
    if not required.issubset(frame.columns):
        raise ValueError("Dataset must contain 'article' and 'summary' columns.")
    prepared = frame[["article", "summary"]].dropna().drop_duplicates().copy()
    prepared["article_clean"] = prepared["article"].map(clean_text)
    prepared["summary_clean"] = prepared["summary"].map(clean_text)
    prepared = prepared[(prepared["article_clean"] != "") & (prepared["summary_clean"] != "")]
    prepared["summary_seq"] = START_TOKEN + " " + prepared["summary_clean"] + " " + END_TOKEN
    prepared["article_len"] = prepared["article_clean"].str.split().map(len)
    prepared["summary_len"] = prepared["summary_clean"].str.split().map(len)
    return prepared.reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic 70/15/15 train, validation, and test splits."""

    train, temporary = train_test_split(frame, test_size=0.30, random_state=seed)
    validation, test = train_test_split(temporary, test_size=0.50, random_state=seed)
    return train, validation, test
