from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot_inference import ChatbotService
from src.config import (
    MODEL_METADATA_PATH, RESPONSIBLE_USE_NOTE, SAMPLE_DATA_PATH, SAMPLE_PROMPTS_PATH,
    SOURCE_TOKENIZER_PATH, TARGET_TOKENIZER_PATH, TOKENIZER_META_PATH, WEIGHTS_PATH,
)

st.set_page_config(page_title="Seq2Seq Attention Chatbot", page_icon="💬", layout="wide")
st.title("💬 Conversational Chatbot with Seq2Seq and Attention")
st.caption(
    "A portfolio demonstration of encoder-decoder LSTM inference, additive attention, "
    "greedy token generation, and responsible chatbot deployment."
)
st.warning(RESPONSIBLE_USE_NOTE)

@st.cache_resource
def load_service():
    conversations = pd.read_csv(SAMPLE_DATA_PATH)
    return ChatbotService(
        WEIGHTS_PATH, SOURCE_TOKENIZER_PATH, TARGET_TOKENIZER_PATH,
        TOKENIZER_META_PATH, conversations
    )

@st.cache_data
def load_assets():
    prompts = json.loads(SAMPLE_PROMPTS_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    conversations = pd.read_csv(SAMPLE_DATA_PATH)
    metrics = json.loads((PROJECT_ROOT/"outputs"/"model_metrics.json").read_text(encoding="utf-8"))
    return prompts, metadata, conversations, metrics

service = load_service()
sample_prompts, metadata, conversations, metrics = load_assets()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "draft_message" not in st.session_state:
    st.session_state.draft_message = ""

with st.sidebar:
    st.header("Chat controls")
    selected_prompt = st.selectbox("Sample prompt", sample_prompts, index=0)
    if st.button("Use sample prompt", use_container_width=True):
        st.session_state.draft_message = selected_prompt
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.last_result = None
        st.rerun()
    st.divider()
    st.subheader("Model scope")
    st.write(f"**Source vocabulary:** {metadata['source_vocab_size']}")
    st.write(f"**Target vocabulary:** {metadata['target_vocab_size']}")
    st.write(f"**Maximum input:** {metadata['max_source_length']} tokens")
    st.write(f"**Maximum response:** {metadata['max_target_length'] - 1} steps")
    st.write(f"**Parameters:** {metadata['parameter_count']:,}")
    st.caption("Inference uses exported NumPy weights; no retraining occurs.")

with st.form("chat_form", clear_on_submit=True):
    user_message = st.text_input(
        "Enter a message",
        key="draft_message",
        placeholder="Example: what should i do next",
    )
    submitted = st.form_submit_button("Send message")

if submitted:
    try:
        result = service.respond(user_message)
        st.session_state.chat_messages.append({"role":"user","content":user_message})
        st.session_state.chat_messages.append({"role":"assistant","content":result.response})
        st.session_state.last_result = result
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

chat_tab, attention_tab, performance_tab, details_tab = st.tabs(
    ["Chat", "Attention and decoding", "Model performance", "Architecture and limitations"]
)

with chat_tab:
    if not st.session_state.chat_messages:
        st.info("Choose a sample prompt or enter a short message to test the model.")
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    result = st.session_state.last_result
    if result is not None:
        a,b,c,d = st.columns(4)
        a.metric("Average token confidence", f"{result.average_confidence:.1%}")
        b.metric("Input OOV ratio", f"{result.oov_ratio:.1%}")
        c.metric("Generated tokens", len(result.generated_tokens))
        d.metric("Fallback used", "Yes" if result.used_fallback else "No")
        if result.used_fallback:
            st.warning(result.fallback_reason)
            with st.expander("Raw model output"):
                st.write(result.raw_model_response or "No usable tokens generated.")
        with st.expander("Retrieval-baseline comparison"):
            st.write(f"**Closest stored prompt:** {result.retrieval.matched_input}")
            st.write(f"**Retrieved response:** {result.retrieval.response}")
            st.write(f"**Token-overlap similarity:** {result.retrieval.similarity:.1%}")

with attention_tab:
    result = st.session_state.last_result
    if result is None:
        st.info("Generate a response first to view token-level decoding details.")
    else:
        st.subheader("Generated-token probabilities")
        probability_frame = pd.DataFrame({
            "token": result.generated_tokens,
            "selected_probability": result.token_confidences,
        })
        if probability_frame.empty:
            st.info("No usable response tokens were generated.")
        else:
            st.dataframe(probability_frame, use_container_width=True)
            fig = px.bar(probability_frame, x="token", y="selected_probability",
                         title="Selected Token Probability by Decoder Step")
            fig.update_yaxes(range=[0,1])
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Additive-attention visualization")
        if result.attention_weights.size == 0 or not result.generated_tokens:
            st.info("No attention matrix is available for this response.")
        else:
            retained = max(len(result.input_tokens), 1)
            matrix = result.attention_weights[:, :retained].astype(float)
            sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)
            heatmap = px.imshow(
                matrix, x=result.input_tokens, y=result.generated_tokens, aspect="auto",
                labels={"x":"Encoder input token","y":"Generated response token","color":"Attention"},
                title="Normalized Attention by Decoder Step"
            )
            st.plotly_chart(heatmap, use_container_width=True)
            st.caption("Attention weights show which input positions were emphasized for each token.")

with performance_tab:
    supplied = metrics["supplied_notebook"]
    replay = metrics["canonical_prompt_replay"]
    st.subheader("Supplied notebook metrics")
    cols = st.columns(4)
    cols[0].metric("Validation BLEU-like", f"{supplied['validation_bleu_like']:.3f}")
    cols[1].metric("Test BLEU-like", f"{supplied['test_bleu_like']:.3f}")
    cols[2].metric("Exact match", f"{supplied['exact_match_ratio']:.1%}")
    cols[3].metric("Final validation loss", f"{supplied['final_validation_loss']:.4f}")
    st.error(metrics["qualification"])
    st.subheader("Canonical prompt replay")
    cols = st.columns(4)
    cols[0].metric("Prompts", replay["prompt_count"])
    cols[1].metric("Exact match", f"{replay['exact_match_ratio']:.1%}")
    cols[2].metric("BLEU-like", f"{replay['bleu_like_mean']:.3f}")
    cols[3].metric("Average confidence", f"{replay['average_token_confidence']:.1%}")
    sample_responses = pd.read_csv(PROJECT_ROOT/"outputs"/"sample_chat_responses.csv")
    st.dataframe(sample_responses[
        ["user_input","reference_response","predicted_response","average_token_confidence"]
    ], use_container_width=True)
    history = pd.read_csv(PROJECT_ROOT/"outputs"/"training_history.csv")
    st.plotly_chart(px.line(history, x="epoch", y=["loss","val_loss"],
                            title="Training and Validation Loss"), use_container_width=True)
    baseline = pd.read_csv(PROJECT_ROOT/"outputs"/"baseline_comparison.csv")
    bfig = px.bar(baseline, x="approach", y="canonical_exact_match",
                  title="Canonical Prompt Baseline Comparison")
    bfig.update_yaxes(range=[0,1.1])
    st.plotly_chart(bfig, use_container_width=True)

with details_tab:
    st.subheader("Architecture")
    st.code("""User message
→ source tokenization and padding
→ 128-dimensional source embedding
→ 128-unit encoder LSTM
→ 128-dimensional target embedding
→ 128-unit decoder LSTM
→ additive attention over encoder outputs
→ context concatenation
→ time-distributed softmax over 81 target tokens""", language="text")
    st.subheader("How inference works")
    st.markdown("""
1. Clean and tokenize the user message.
2. Run the encoder once to obtain sequence outputs and final states.
3. Start the decoder with the `sostok` token.
4. Generate one token at a time using greedy decoding.
5. Use additive attention at every decoder step.
6. Stop at `eostok` or the maximum response length.
7. Use a responsible fallback for strongly out-of-domain input.
""")
    st.subheader("Known limitations")
    st.markdown("""
- The model was trained on only 20 fixed synthetic dialogue templates.
- The original row split placed every exact pair in every split.
- Perfect notebook metrics therefore reflect memorization, not open-domain ability.
- The chatbot has no persistent long-term conversational memory.
- Unseen words and paraphrases may trigger a fallback or poor response.
- This educational LSTM system is not a Transformer or LLM replacement.
""")
    st.warning(RESPONSIBLE_USE_NOTE)
