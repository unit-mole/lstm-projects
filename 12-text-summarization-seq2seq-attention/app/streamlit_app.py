"""Streamlit demo for abstractive text summarization with Seq2Seq attention."""

from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MAX_BATCH_ROWS, SAMPLE_ARTICLES_PATH
from src.model_evaluation import compute_rouge
from src.summarization_inference import Summarizer
from src.visualization import create_attention_heatmap

st.set_page_config(
    page_title="Seq2Seq Attention Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1.5rem; padding-bottom: 3rem;}
.hero {padding: 1.35rem 1.5rem; border-radius: 14px; background: linear-gradient(135deg,#17365D,#2F75B5); color:white; margin-bottom:1rem;}
.hero h1 {margin:0 0 .35rem 0; font-size:2.05rem;}
.hero p {margin:0; opacity:.95;}
.summary-card {padding:1.1rem 1.2rem; border-left:5px solid #2F75B5; border-radius:8px; background:#F4F8FC; font-size:1.08rem;}
.notice {padding:.9rem 1rem; border-radius:8px; background:#FFF4E5; border:1px solid #F0C36D;}
.small-note {font-size:.9rem; color:#52606D;}
[data-testid="stMetricValue"] {font-size:1.4rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the pretrained encoder and decoder models...")
def load_summarizer() -> Summarizer:
    return Summarizer()


@st.cache_data
def load_samples() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_ARTICLES_PATH)


def render_result(result, reference_summary: str | None = None) -> None:
    st.markdown("### Generated summary")
    st.markdown(
        f'<div class="summary-card">{result.summary}</div>',
        unsafe_allow_html=True,
    )
    if result.warning:
        st.warning(result.warning)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input words", result.input_word_count)
    col2.metric("Summary words", result.summary_word_count)
    col3.metric("Compression", f"{result.compression_ratio:.1%}")
    col4.metric("Out-of-vocabulary", f"{result.oov_ratio:.1%}")

    if reference_summary:
        st.markdown("### Reference comparison")
        st.write(reference_summary)
        rouge = compute_rouge(reference_summary.lower().strip(" ."), result.summary)
        r1, r2, rl = st.columns(3)
        r1.metric("ROUGE-1 F1", f"{rouge['rouge_1_f1']:.3f}")
        r2.metric("ROUGE-2 F1", f"{rouge['rouge_2_f1']:.3f}")
        rl.metric("ROUGE-L F1", f"{rouge['rouge_l_f1']:.3f}")

    if (
        result.attention_matrix is not None
        and result.generated_tokens
        and result.source_tokens
    ):
        with st.expander("View additive-attention alignment", expanded=False):
            figure = create_attention_heatmap(
                result.attention_matrix,
                result.source_tokens,
                result.generated_tokens,
            )
            st.pyplot(figure, clear_figure=True, use_container_width=True)
            st.caption(
                "The heatmap shows decoder alignment weights over input tokens. "
                "It is useful for inspection but is not a causal explanation."
            )


summarizer = load_summarizer()
samples = load_samples()

st.markdown(
    """
<div class="hero">
  <h1>Text Summarization using Seq2Seq with Attention</h1>
  <p>Generate concise abstractive summaries with an Encoder-Decoder LSTM and additive attention.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="notice"><strong>Responsible-use notice:</strong> This is an educational portfolio demo trained on a small deterministic synthetic corpus. Generated summaries may omit context, repeat patterns, or be inaccurate. Do not use it for legal, medical, financial, safety-critical, confidential, or official documents without human review.</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model controls")
    decoding = st.radio(
        "Decoding method",
        ["Greedy", "Beam search"],
        help="Greedy selects the highest-probability token. Beam search retains multiple candidate sequences.",
    )
    beam_width = 3
    if decoding == "Beam search":
        beam_width = st.slider("Beam width", 2, 5, 3)
    st.divider()
    st.subheader("Verified artifacts")
    metadata = summarizer.metadata
    st.write(f"**Source vocabulary:** {metadata['sequence']['source_vocab_size']}")
    st.write(f"**Target vocabulary:** {metadata['sequence']['target_vocab_size']}")
    st.write(f"**Maximum input:** {metadata['sequence']['max_source_length']} tokens")
    st.write(f"**Maximum target:** {metadata['sequence']['max_target_length']} tokens")
    st.write(f"**Training records:** {metadata['data']['records']:,}")
    st.divider()
    st.link_button(
        "View GitHub project",
        "https://github.com/unit-mole/lstm-projects/tree/main/12-text-summarization-seq2seq-attention",
        use_container_width=True,
    )

sample_tab, manual_tab, batch_tab = st.tabs(
    ["Preloaded samples", "Manual text", "CSV batch summarization"]
)

with sample_tab:
    st.subheader("Recruiter-friendly sample workflow")
    selected_title = st.selectbox("Select a sample article", samples["title"].tolist())
    selected = samples.loc[samples["title"] == selected_title].iloc[0]
    st.text_area("Input article", selected["input_text"], height=190, disabled=True)
    if st.button("Generate sample summary", type="primary", key="sample_generate"):
        with st.spinner("Generating summary..."):
            result = summarizer.summarize(
                selected["input_text"],
                decoding_method=decoding,
                beam_width=beam_width,
                include_attention=decoding == "Greedy",
            )
        render_result(result, selected["target_summary"])

with manual_tab:
    st.subheader("Summarize your own text")
    manual_text = st.text_area(
        "Paste a document or paragraph",
        height=240,
        placeholder=(
            "Paste at least eight words. The model works best on content similar "
            "to the synthetic organization/action/theme/impact training domain."
        ),
    )
    st.caption(
        "Avoid private, confidential, copyrighted, or personally identifiable text. "
        "Inputs are processed only for the active session by this application."
    )
    if st.button("Generate manual summary", type="primary", key="manual_generate"):
        try:
            with st.spinner("Generating summary..."):
                result = summarizer.summarize(
                    manual_text,
                    decoding_method=decoding,
                    beam_width=beam_width,
                    include_attention=decoding == "Greedy",
                )
            render_result(result)
        except ValueError as exc:
            st.error(str(exc))

with batch_tab:
    st.subheader("Batch summarization from CSV")
    st.write(
        "Upload a CSV, choose the input-text column, and generate up to "
        f"{MAX_BATCH_ROWS} summaries per run."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Use `data/sample_batch.csv` to test this workflow.")
    else:
        try:
            batch_frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"The CSV could not be read: {exc}")
            batch_frame = pd.DataFrame()
        if not batch_frame.empty:
            st.dataframe(batch_frame.head(10), use_container_width=True)
            text_column = st.selectbox("Input text column", batch_frame.columns.tolist())
            reference_options = ["None"] + batch_frame.columns.tolist()
            reference_column = st.selectbox(
                "Reference summary column (optional)", reference_options
            )
            row_limit = st.number_input(
                "Rows to process",
                min_value=1,
                max_value=min(MAX_BATCH_ROWS, len(batch_frame)),
                value=min(20, len(batch_frame)),
            )
            if st.button("Generate batch summaries", type="primary"):
                selected_frame = batch_frame.head(int(row_limit)).copy()
                progress = st.progress(0, text="Generating summaries...")
                records = []
                values = selected_frame[text_column].fillna("").astype(str).tolist()
                for index, value in enumerate(values):
                    try:
                        result = summarizer.summarize(
                            value,
                            decoding_method=decoding,
                            beam_width=beam_width,
                            include_attention=False,
                        )
                        record = result.as_record()
                    except ValueError as exc:
                        record = {
                            "input_text": value,
                            "generated_summary": "",
                            "decoding_method": decoding,
                            "input_word_count": len(value.split()),
                            "summary_word_count": 0,
                            "compression_ratio": 0.0,
                            "oov_ratio": 0.0,
                            "truncated": False,
                            "warning": str(exc),
                        }
                    if reference_column != "None":
                        reference = str(selected_frame.iloc[index][reference_column])
                        record.update(compute_rouge(reference.lower().strip(" ."), record["generated_summary"]))
                    records.append(record)
                    progress.progress((index + 1) / len(values), text="Generating summaries...")
                progress.empty()
                output_frame = pd.DataFrame(records)
                st.success(f"Generated {len(output_frame)} row-level results.")
                st.dataframe(output_frame, use_container_width=True)
                csv_bytes = output_frame.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download summarized CSV",
                    data=csv_bytes,
                    file_name="generated_summaries.csv",
                    mime="text/csv",
                )

st.divider()
with st.expander("How the model works"):
    st.markdown(
        """
1. Input text is cleaned and converted to a maximum of **49 source tokens**.
2. The encoder embedding and LSTM create token-level outputs plus hidden and cell states.
3. The decoder LSTM generates one summary token at a time.
4. Additive attention aligns each decoder step with relevant encoder positions.
5. A softmax layer selects from the **57-token target vocabulary** until the end token or length limit is reached.
        """
    )

with st.expander("Limitations"):
    st.markdown(
        """
- The model was trained on a small templated synthetic corpus and is not a general-purpose summarizer.
- Out-of-domain terms map to an unknown token and can produce generic or incorrect summaries.
- Inputs beyond 49 cleaned tokens are truncated.
- Greedy and beam decoding do not guarantee factual consistency.
- Modern Transformer and large-language-model summarizers generally handle broader language and longer context more effectively.
        """
    )
