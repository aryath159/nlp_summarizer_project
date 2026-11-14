import os
import requests
import pandas as pd
import streamlit as st

# Replace API_URL = "http://localhost:8000" at the top of frontend/app.py with:
with st.sidebar:
    st.title("🌐 Cloud GPU Connection")
    API_URL = st.text_input(
        "Kaggle FastAPI Tunnel URL:",
        value="http://localhost:8000",
        help="https://sip-edgy-lens.ngrok-free.dev"
    ).rstrip("/")

st.set_page_config(
    page_title="Dialogue Summarizer | QLoRA + DPO",
    page_icon="🧠",
    layout="wide",
)

PRESET_EXAMPLES = {
    "Select an example...": "",
    "Example 1: Office Instant Messaging Memo (Role-Swap Test Case)": (
        "#Person1#: Ms. Dawson, I need you to take a dictation for me.\n"
        "#Person2#: Yes, sir...\n"
        "#Person1#: This should go out as an intra-office memorandum to all employees by this afternoon. Are you ready?\n"
        "#Person2#: Yes, sir. Go ahead.\n"
        "#Person1#: Attention all staff... Effective immediately, all office communications are restricted to email correspondence and official memos. The use of Instant Message programs by employees during working hours is strictly prohibited.\n"
        "#Person2#: Sir, does this apply to intra-office communications only? Or will it also restrict external communications?\n"
        "#Person1#: It should apply to all communications, not only in this office between employees, but also any outside communications.\n"
        "#Person2#: But sir, many employees use Instant Messaging to communicate with their clients.\n"
        "#Person1#: They will just have to change their communication methods. I don't want any - one using Instant Messaging in this office. It wastes too much time! Now, please continue with the memo. Where were we?\n"
        "#Person2#: This applies to internal and external communications.\n"
        "#Person1#: Yes. Any employee who persists in using Instant Messaging will first receive a warning and be placed on probation. At second offense, the employee will face termination. Any questions regarding this new policy may be directed to department heads.\n"
        "#Person2#: Is that all?\n"
        "#Person1#: Yes. Please get this memo typed up and distributed to all employees before 4 pm."
    ),
    "Example 2: Product Launch & Backend Deployment": (
        "#Person1#: Have we finalized the microservice migration before Monday's client demo?\n"
        "#Person2#: The PostgreSQL database migration is complete, but Sarah is still load-testing the Redis cache layer.\n"
        "#Person1#: When will Sarah finish the benchmarks?\n"
        "#Person2#: She promised to send the latency report by 5 PM today. If p99 latency stays under 40ms, we will deploy to production tomorrow morning.\n"
        "#Person1#: Great. Please schedule a 15-minute sync with Sarah and DevOps at 5:30 PM to review the numbers."
    ),
}

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ Inference Controls")
    max_tokens = st.slider("Max Summary Tokens", min_value=40, max_value=250, value=120, step=10)
    use_hierarchical = st.toggle(
        "Hierarchical Map-Reduce Mode",
        value=False,
        help="Splits long transcripts (>600 tokens) into overlapping chunks, summarizes each, and synthesizes a master summary.",
    )
    top_k_extractive = st.slider("TextRank Sentences (Baseline)", 1, 5, 2)
    st.divider()
    st.markdown(
        "**Architecture Stack:**\n"
        "- **Base Model:** `Qwen/Qwen2.5-3B-Instruct`\n"
        "- **Fine-Tuning:** 4-Bit NF4 QLoRA ($r=16, \\alpha=32$, Attn + MLP)\n"
        "- **Alignment:** Direct Preference Optimization (DPO)\n"
        "- **Dataset:** `DialogSum` (12.4k dialogues)"
    )

st.title("🧠 Abstractive Dialogue Summarization & Action Extraction")
st.caption("NLP Semester Project — Comparing Classical Extractive, Zero-Shot Causal LM, and QLoRA + DPO Alignment")

tab1, tab2, tab3 = st.tabs([
    "✨ Interactive Summarizer & JSON Extractor",
    "⚔️ Multi-Paradigm Model Arena (Side-by-Side)",
    "📊 Evaluation Benchmarks & Training Curves",
])

# ---------------------------------------------------------
# TAB 1: Interactive Summarizer & Structured JSON
# ---------------------------------------------------------
with tab1:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("📝 Input Dialogue Transcript")
        selected_preset = st.selectbox("Load Preset Test Dialogue:", list(PRESET_EXAMPLES.keys()))
        uploaded_file = st.file_uploader("Or upload a .txt transcript file:", type=["txt"])

        default_text = PRESET_EXAMPLES[selected_preset]
        if uploaded_file is not None:
            default_text = uploaded_file.read().decode("utf-8")

        dialogue_input = st.text_area(
            "Conversation Text:",
            value=default_text,
            height=320,
            placeholder="#Person1#: Hello...\n#Person2#: Hi there...",
        )
        run_btn = st.button("🚀 Generate Summary & Extract Action Items", type="primary", use_container_width=True)

    with col_out:
        st.subheader("🎯 Model Output")
        if run_btn:
            if not dialogue_input.strip():
                st.warning("Please enter or select a dialogue transcript first.")
            else:
                with st.spinner("Running QLoRA + DPO inference..."):
                    resp = requests.post(
                        f"{API_URL}/compare-all",
                        json={
                            "dialogue": dialogue_input,
                            "max_new_tokens": max_tokens,
                            "use_hierarchical": use_hierarchical,
                            "top_k_extractive": top_k_extractive,
                        },
                    ).json()

                ft_data = resp["finetuned_model"]
                st.success(ft_data["summary"])

                # Metrics bar
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{ft_data.get('latency_sec', 0)} s")
                if "compression_ratio" in ft_data:
                    m2.metric("Output Tokens", ft_data["output_tokens"])
                    m3.metric("Compression Ratio", f"{int(ft_data['compression_ratio']*100)}%")
                elif "num_chunks" in ft_data:
                    m2.metric("Hierarchical Chunks", ft_data["num_chunks"])

                if use_hierarchical and "chunk_summaries" in ft_data:
                    with st.expander("🔍 View Intermediate Map-Step Chunk Summaries"):
                        for idx, c_sum in enumerate(ft_data["chunk_summaries"]):
                            st.markdown(f"**Chunk {idx+1}:** {c_sum}")

                st.subheader("🗂️ Structured JSON & Action Items")
                st.json(resp["structured_json"])

# ---------------------------------------------------------
# TAB 2: Multi-Paradigm Model Arena
# ---------------------------------------------------------
with tab2:
    st.subheader("⚔️ Side-by-Side Paradigm Comparison")
    st.markdown(
        "Compare how **Classical Unsupervised Graph NLP (TextRank)**, **Zero-Shot Base LLM**, and our "
        "**Fine-Tuned QLoRA + DPO Model** handle the exact same conversation."
    )
    arena_btn = st.button("⚡ Run 3-Model Comparison on Current Input", use_container_width=True)

    if arena_btn and dialogue_input.strip():
        with st.spinner("Evaluating all three paradigms..."):
            resp = requests.post(
                f"{API_URL}/compare-all",
                json={
                    "dialogue": dialogue_input,
                    "max_new_tokens": max_tokens,
                    "use_hierarchical": use_hierarchical,
                    "top_k_extractive": top_k_extractive,
                },
            ).json()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("📊 **1. Classical TextRank (Extractive)**")
            st.write(resp["extractive_summary"])
            st.caption("Selects verbatim sentences using TF-IDF + PageRank centrality. Cannot paraphrase or synthesize.")

        with c2:
            st.warning("🤖 **2. Zero-Shot Base Model (`Qwen2.5-3B`)**")
            st.write(resp["base_model"]["summary"])
            st.caption(f"Latency: {resp['base_model']['latency_sec']}s | Often verbose or prone to speaker role confusion.")

        with c3:
            st.success("🏆 **3. Proposed QLoRA (Attn+MLP) + DPO**")
            st.write(resp["finetuned_model"]["summary"])
            st.caption(f"Latency: {resp['finetuned_model']['latency_sec']}s | Concise DialogSum style with DPO role-swap alignment.")

# ---------------------------------------------------------
# TAB 3: Saved Kaggle Evaluation Benchmarks & Plots
# ---------------------------------------------------------
with tab3:
    st.subheader("📈 Quantitative Ablation & Baseline Results")
    csv_path = "./artifacts/model_comparison_results.csv"
    plot_path = "./artifacts/training_perplexity_curves.png"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        st.dataframe(df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)
    else:
        st.info(f"Place `model_comparison_results.csv` inside `./artifacts/` to display the benchmark table.")

    st.divider()
    st.subheader("📉 SFT Cross-Entropy Loss & Validation Perplexity ($PPL = \\exp(\\mathcal{L})$)")
    if os.path.exists(plot_path):
        st.image(plot_path, use_container_width=True)
    else:
        st.info(f"Place `training_perplexity_curves.png` inside `./artifacts/` to display the training curves.")


#  source .venv/bin/activate
#   streamlit run frontend/app.py