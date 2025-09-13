import os, json, time, io
import numpy as np
import pandas as pd
import streamlit as st
import boto3
import faiss

# =========================
# Bedrock clients (use SageMaker IAM role, no keys in code)
# =========================
def bedrock_runtime(region: str):
    # If SageMaker Studio / Notebook has an attached role, boto3 picks it up automatically.
    return boto3.client("bedrock-runtime", region_name=region)

# Default region (change if your models are enabled elsewhere)
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Model IDs you likely have (adjust to what your account enables)
DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"  # Titan Embeddings v2
DEFAULT_LLM_MODEL   = "anthropic.claude-3-haiku-20240307-v1:0"  # fast/cheap; or Sonnet if allowed

# =========================
# Streamlit page setup + CSS (ChatGPT-ish)
# =========================
st.set_page_config(page_title="RAG Chat • Bedrock + SageMaker", layout="wide")
st.markdown("""
<style>
.stStatusWidget, .stDeployButton, .stToolbar { display: none !important; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
.stApp > header { background-color: #87CEFA; color: white; }
.result-card {
  background: #ffffff; border-radius: 12px; padding: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,.08); border: 1px solid #e5e7eb;
  margin: 8px 0 16px;
}
.stCodeBlock code { max-width:100%; white-space:pre-wrap; background:#f3f4f6; }
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar: Settings & Controls
# =========================
st.sidebar.title("⚙️ Settings")
bedrock_region = st.sidebar.text_input("AWS Region", value=DEFAULT_REGION, help="Region with Bedrock model access")
embed_model_id = st.sidebar.text_input("Embeddings Model ID", value=DEFAULT_EMBED_MODEL)
llm_model_id   = st.sidebar.text_input("Chat Model ID", value=DEFAULT_LLM_MODEL)
sim_threshold  = st.sidebar.slider("Similarity threshold (cosine)", 0.0, 1.0, 0.30, 0.01)
top_k          = st.sidebar.slider("Top-k candidates (retrieval)", 1, 10, 5, 1)
temperature    = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.0, 0.1)
allow_stream   = st.sidebar.checkbox("Stream responses (when supported)", True)

brt = bedrock_runtime(bedrock_region)

st.title("🏦 Bedrock RAG Chat (SageMaker POC)")
st.caption("Embeddings: Titan • LLM: Claude via Bedrock • Retrieval: FAISS • Guardrails: threshold + top-1 by default")

# =========================
# Load/Upload Knowledge Base (CSV)
# Required columns: MetricName, Definition, SQL/SourceTable, Derivation
# =========================
DEFAULT_KB = pd.DataFrame([
    {
        "MetricName":"LTV Ratio",
        "Definition":"Loan-to-Value ratio = Loan Amount / Appraised Property Value",
        "SQL/SourceTable":"SELECT loan_amount/appraised_value FROM mortgage_loans",
        "Derivation":"LTV = Loan Amount ÷ Appraised Property Value",
    },
    {
        "MetricName":"DTI Ratio",
        "Definition":"Debt-to-Income ratio = Borrower Monthly Debt / Monthly Income",
        "SQL/SourceTable":"SELECT monthly_debt/monthly_income FROM borrowers",
        "Derivation":"DTI = Monthly Debt ÷ Monthly Income",
    }
])

st.subheader("📚 Knowledge Base")
kb_file = st.file_uploader("Upload CSV (required columns: MetricName, Definition, SQL/SourceTable, Derivation)", type=["csv"])
if kb_file:
    kb_df = pd.read_csv(kb_file)
else:
    kb_df = DEFAULT_KB.copy()

required_cols = {"MetricName","Definition","SQL/SourceTable","Derivation"}
missing = required_cols - set(kb_df.columns)
if missing:
    st.error(f"KB is missing columns: {missing}")
    st.stop()

# show sample
with st.expander("Preview KB"):
    st.dataframe(kb_df.head(20), use_container_width=True)

# =========================
# Build chunks from KB
# =========================
def build_chunks(df: pd.DataFrame):
    # one chunk per metric row
    return [
        f"Metric: {row['MetricName']}\n"
        f"Definition: {row['Definition']}\n"
        f"Derivation: {row['Derivation']}\n"
        f"SQL: {row['SQL/SourceTable']}"
        for _, row in df.iterrows()
    ]

# =========================
# Embeddings via Bedrock (Titan)
# =========================
def titan_embed_one(text: str) -> np.ndarray:
    body = {"inputText": text}
    resp = brt.invoke_model(
        modelId=embed_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    out = json.loads(resp["body"].read())
    return np.array(out["embedding"], dtype="float32")

def titan_embed_many(texts) -> np.ndarray:
    vecs = []
    for t in texts:
        vecs.append(titan_embed_one(t))
    return np.vstack(vecs).astype("float32")

# =========================
# Index management (FAISS in-memory) + caching to avoid re-embedding
# =========================
@st.cache_resource(show_spinner=True)
def build_index(df: pd.DataFrame, kb_hash: str):
    chunks = build_chunks(df)
    embs = titan_embed_many(chunks)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs_norm = embs / norms
    dim = embs_norm.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs_norm.astype("float32"))
    return chunks, index, embs_norm

# Create a tiny hash of KB to know when to rebuild
kb_hash = str(hash(pd.util.hash_pandas_object(kb_df, index=False).sum()))
rebuild = st.button("🔁 Rebuild Embeddings")
if rebuild:
    st.cache_resource.clear()

with st.spinner("Building embeddings index..."):
    chunks, index, embs_norm = build_index(kb_df, kb_hash)

# =========================
# Retrieval helpers
# =========================
def cosine_sims(q_vec: np.ndarray, mat_norm: np.ndarray) -> np.ndarray:
    q = q_vec / (np.linalg.norm(q_vec) + 1e-12)
    return (mat_norm @ q).astype("float32")

def retrieve_best(query: str, k: int, threshold: float):
    q_vec = titan_embed_one(query)
    _, I = index.search(q_vec[np.newaxis, :], k=k)
    cand_ix = I[0].tolist()
    sims = cosine_sims(q_vec, embs_norm[cand_ix, :])
    # sort by similarity
    order = np.argsort(-sims)
    cand_ix = [cand_ix[i] for i in order]
    sims = [float(sims[i]) for i in order]
    # keep above threshold
    keep = [(cand_ix[i], sims[i]) for i in range(len(cand_ix)) if sims[i] >= threshold]
    return keep, q_vec

# =========================
# Bedrock: Claude chat
# =========================
STRICT_SYSTEM = (
    "You are an assistant for mortgage risk metrics. "
    "Only answer if the provided context contains relevant information. "
    "If the context does not contain the answer, reply exactly: "
    "'I don't know — please check the KB.' Do NOT infer or guess."
)

def claude_chat(user_q: str, context: str, stream: bool = True) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": STRICT_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"User question: {user_q}\n\n"
                            f"Context:\n{context}\n\n"
                            "Answer in clear business language and show the derivation and SQL sources you used."
                        )
                    }
                ],
            }
        ],
        "max_tokens": 700,
        "temperature": temperature,
    }

    if stream:
        try:
            resp = brt.invoke_model_with_response_stream(
                modelId=llm_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            collected = ""
            for event in resp.get("body"):
                if "chunk" in event:
                    data = json.loads(event["chunk"]["bytes"])
                    # Anthropic stream chunks contain "content" parts
                    for part in data.get("content", []):
                        if part.get("type") == "text":
                            collected += part.get("text", "")
                            yield collected  # incremental
            return
        except Exception:
            pass  # fall back to non-streaming

    # Non-streaming
    resp = brt.invoke_model(
        modelId=llm_model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    out = json.loads(resp["body"].read())
    text = "".join([p.get("text", "") for p in out.get("content", []) if p.get("type") == "text"])
    yield text

# =========================
# Chat state
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_ctx" not in st.session_state:
    st.session_state.last_ctx = None
if "last_q" not in st.session_state:
    st.session_state.last_q = None

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# Chat input + RAG flow
# =========================
user_q = st.chat_input("Ask about a metric (e.g., 'How is LTV calculated?')")

# Regenerate / Redo controls
cols = st.columns(3)
with cols[0]:
    do_regen = st.button("🔄 Regenerate")
with cols[1]:
    ask_clarify = st.button("❓ Ask to clarify")
with cols[2]:
    show_sources = st.button("📂 Show last sources")

def answer_with_context(question: str):
    keep, _ = retrieve_best(question, k=top_k, threshold=sim_threshold)
    if not keep:
        return None, "I don't know — your question doesn’t match anything in the KB. Try naming a specific metric (e.g., **LTV**, **DTI**).", []
    # Use top-1 by default
    best_idx, best_sim = keep[0]
    ctx = chunks[best_idx]
    st.session_state.last_ctx = [chunks[i] for (i, s) in keep[:3]]  # keep up to 3 for source preview
    st.session_state.last_q = question
    return ctx, None, keep

if user_q:
    st.session_state.chat.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    context, err, keep = answer_with_context(user_q)

    with st.chat_message("assistant"):
        if err:
            st.markdown(err)
            st.session_state.chat.append({"role": "assistant", "content": err})
        else:
            # Stream (or fallback) the LLM output
            placeholder = st.empty()
            final_text = ""
            for partial in claude_chat(user_q, context, stream=allow_stream):
                final_text = partial
                placeholder.markdown(final_text)
            st.session_state.chat.append({"role": "assistant", "content": final_text})

            with st.expander("📂 Retrieved Source(s)"):
                for i, (idx, sim) in enumerate(keep[:3], start=1):
                    st.markdown(f"**Match {i}** • cosine≈{sim:.2f}")
                    st.code(chunks[idx])

# Redo / Regenerate uses last question & context
if do_regen and st.session_state.get("last_q") and st.session_state.get("last_ctx"):
    with st.chat_message("assistant"):
        context = st.session_state.last_ctx[0]
        placeholder = st.empty()
        final_text = ""
        for partial in claude_chat(st.session_state.last_q, context, stream=allow_stream):
            final_text = partial
            placeholder.markdown(final_text)
        st.session_state.chat.append({"role": "assistant", "content": final_text})

if ask_clarify and st.session_state.get("last_q"):
    with st.chat_message("assistant"):
        msg = f"Could you clarify what you’re looking for about **{st.session_state.last_q}**? For example, the derivation, the SQL source, or a business explanation?"
        st.markdown(msg)
        st.session_state.chat.append({"role": "assistant", "content": msg})

if show_sources and st.session_state.get("last_ctx"):
    with st.chat_message("assistant"):
        st.markdown("Here are the most recent retrieved sources:")
        for i, src in enumerate(st.session_state.last_ctx, start=1):
            st.markdown(f"**Source {i}**")
            st.code(src)
        st.session_state.chat.append({"role": "assistant", "content": "Shown the retrieved sources."})

# Feedback (simple thumbs)
st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    if st.button("👍 Helpful"):
        st.success("Thanks! Logged as helpful.")
with c2:
    if st.button("👎 Not helpful"):
        st.warning("Thanks! Logged as not helpful.")