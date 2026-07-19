from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fake_news_pipeline import FakeNewsPipeline

MODEL_DIRECTORY = PROJECT_ROOT / "models"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample_news.csv"

st.set_page_config(
    page_title="Fake News Detection — LSTM Portfolio Demo",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the LSTM checkpoint...")
def load_pipeline() -> FakeNewsPipeline:
    return FakeNewsPipeline.load(MODEL_DIRECTORY)


@st.cache_data
def load_samples() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def responsible_notice() -> None:
    st.warning(
        "**Educational portfolio demo — not a fact checker.** The model detects language "
        "patterns learned from LIAR political claims and may be wrong or biased. Verify claims "
        "through reliable sources, evidence, context, and human review. Do not use this result "
        "for legal, political, journalistic, financial, medical, or public-safety decisions."
    )


def render_result(result) -> None:
    label_column, probability_column, confidence_column = st.columns(3)
    label_column.metric("Predicted label", result.predicted_label)
    probability_column.metric("Fake-news probability", f"{result.fake_probability:.1%}")
    confidence_column.metric("Model confidence", f"{result.confidence:.1%}")

    st.progress(min(max(result.fake_probability, 0.0), 1.0), text="Estimated fake probability")
    st.info(result.interpretation)

    details = st.columns(4)
    details[0].metric("Threshold", f"{result.threshold:.2f}")
    details[1].metric("Tokens used", result.used_token_count)
    details[2].metric("OOV ratio", f"{result.oov_ratio:.1%}")
    details[3].metric("Truncated", "Yes" if result.truncated else "No")

    if result.warning:
        st.warning(result.warning)
    st.caption("Important: confidence is not factual certainty and is not externally calibrated.")


def detect_text_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["text", "statement", "headline", "title", "article", "content", "news"]
    matching = [column for column in preferred if column in frame.columns]
    remaining = [
        column
        for column in frame.columns
        if column not in matching and (frame[column].dtype == "object" or pd.api.types.is_string_dtype(frame[column]))
    ]
    return matching + remaining


pipeline = load_pipeline()
samples = load_samples()
metadata = pipeline.metadata

st.title("🧭 Fake News Detection using Bidirectional LSTM")
st.write(
    "A recruiter-friendly NLP demo that estimates whether a **short political claim or headline** "
    "resembles LIAR examples grouped as fake or real."
)
responsible_notice()

with st.sidebar:
    st.header("Project details")
    st.write("**Dataset:** LIAR short political claims")
    st.write("**Model:** Bidirectional LSTM with masked mean/max pooling")
    st.write(f"**Decision threshold:** {pipeline.threshold:.2f}")
    test_metrics = metadata.get("test_metrics", {})
    st.write(f"**Test accuracy:** {test_metrics.get('accuracy', 0):.1%}")
    st.write(f"**Test ROC-AUC:** {test_metrics.get('roc_auc', 0):.3f}")
    st.write("**GitHub:** Replace with your repository link")
    st.divider()
    st.caption("Best suited to short English-language claims similar to the training domain.")

manual_tab, batch_tab, details_tab = st.tabs(
    ["Manual prediction", "Batch CSV prediction", "Model details & limitations"]
)

with manual_tab:
    st.subheader("Analyze one claim or headline")
    sample_options = ["Write my own text"] + [
        f"{row.sample_type}: {row.text[:70]}..." for row in samples.itertuples()
    ]
    selected = st.selectbox("Start with an illustrative sample", sample_options)
    default_text = ""
    if selected != "Write my own text":
        sample_index = sample_options.index(selected) - 1
        default_text = str(samples.iloc[sample_index]["text"])
        st.caption(str(samples.iloc[sample_index]["notes"]))

    text = st.text_area(
        "News claim, headline, or short statement",
        value=default_text,
        height=180,
        max_chars=8_000,
        placeholder="Paste a short claim or headline here...",
    )
    if st.button("Generate prediction", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Enter text before generating a prediction.")
        else:
            st.markdown("#### Input preview")
            st.code(text.strip())
            render_result(pipeline.predict(text))

with batch_tab:
    st.subheader("Score multiple rows from CSV")
    st.write(
        "Upload a CSV containing a text-like column such as `text`, `statement`, `headline`, "
        "`title`, `article`, or `content`."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            input_frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
        else:
            st.dataframe(input_frame.head(20), use_container_width=True)
            text_columns = detect_text_columns(input_frame)
            if not text_columns:
                st.error("No text-like column was detected.")
            else:
                selected_column = st.selectbox("Text column", text_columns)
                row_limit = st.number_input(
                    "Maximum rows to score",
                    min_value=1,
                    max_value=10_000,
                    value=min(len(input_frame), 1_000),
                    step=100,
                )
                if st.button("Run batch prediction", type="primary", use_container_width=True):
                    subset = input_frame.head(int(row_limit)).copy()
                    scored = pipeline.predict_batch(subset[selected_column].fillna(""))
                    final = pd.concat([subset.reset_index(drop=True), scored.drop(columns="text")], axis=1)

                    st.markdown("#### Scored results")
                    st.dataframe(final, use_container_width=True)
                    distribution = final["predicted_label"].value_counts().rename_axis("label")
                    st.bar_chart(distribution)

                    csv_bytes = final.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download scored CSV",
                        data=csv_bytes,
                        file_name="fake_news_scored_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.caption(
                        "Batch results are model-pattern estimates only and require independent verification."
                    )

with details_tab:
    st.subheader("Model card")
    st.json(
        {
            "scope": metadata.get("task_scope"),
            "positive_class": "Fake",
            "threshold": pipeline.threshold,
            "model_config": metadata.get("model_config"),
            "tokenizer_config": metadata.get("tokenizer_config"),
            "test_metrics": metadata.get("test_metrics"),
            "baseline_test_metrics": metadata.get("baseline_test_metrics"),
        },
        expanded=False,
    )

    st.markdown(
        """
        ### What this model can do
        - Demonstrate sequence preprocessing, LSTM inference, probability output, and batch scoring.
        - Identify linguistic patterns similar to LIAR's labeled political statements.
        - Support educational discussion about precision, recall, false positives, and false negatives.

        ### What this model cannot do
        - Retrieve evidence or verify sources.
        - Determine objective truth from text alone.
        - Reliably score long articles, breaking news, non-political domains, or non-English content.
        - Replace professional fact-checking, journalism, or human review.

        ### Why the baseline matters
        The TF-IDF logistic-regression baseline slightly outperformed the LSTM on the held-out test set. The project reports that result rather than claiming deep learning is automatically superior.
        """
    )
    responsible_notice()
