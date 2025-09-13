import os, json, boto3, numpy as np

REGION = os.environ.get("AWS_REGION", "us-east-1")
EMBED_MODEL = "amazon.titan-embed-text-v2:0"            # adjust if needed
CLAUDE_MODEL = "anthropic.claude-3-haiku-20240307-v1:0" # or a Claude model your acct allows

brt = boto3.client("bedrock-runtime", region_name=REGION)

# --- Titan Embedding test ---
txt = "Loan-to-Value ratio = Loan Amount / Appraised Property Value"
emb_resp = brt.invoke_model(
    modelId=EMBED_MODEL,
    contentType="application/json",
    accept="application/json",
    body=json.dumps({"inputText": txt}),
)
emb_out = json.loads(emb_resp["body"].read())
vec = np.array(emb_out["embedding"], dtype="float32")
print("✅ Titan OK — embedding length:", vec.shape)

# --- Claude chat test ---
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "system": "You are helpful. Answer briefly.",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "2+2?"}]}],
    "max_tokens": 20,
    "temperature": 0.0,
}
chat_resp = brt.invoke_model(
    modelId=CLAUDE_MODEL, contentType="application/json", accept="application/json", body=json.dumps(body)
)
chat_out = json.loads(chat_resp["body"].read())
ans = "".join([p.get("text","") for p in chat_out.get("content",[]) if p.get("type")=="text"])
print("✅ Claude OK — answer:", ans)


import pandas as pd

# If you already have your CSV, set the path here:
CSV_PATH = "mortgage_metrics_kb_extended_with_derivation.csv"

df = pd.read_csv(CSV_PATH)
required = {"MetricName","Definition","SQL/SourceTable","Derivation"}
missing = required - set(df.columns)
assert not missing, f"CSV missing columns: {missing}"
df.head()





best_idx = I[0][0]
best_sim = sims[0]
print("Best sim:", best_sim)

STRICT_SYSTEM = (
    "You are an assistant for mortgage risk metrics. "
    "Only answer if the provided context contains relevant information. "
    "If the context does not contain the answer, reply exactly: "
    "'I don't know — please check the KB.' Do NOT infer or guess."
)

context = chunks[best_idx]
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "system": STRICT_SYSTEM,
    "messages": [{
        "role":"user",
        "content":[{"type":"text","text":
            f"User question: {query}\n\nContext:\n{context}\n\n"
            "Answer in clear business language and show the derivation and SQL sources you used."
        }]
    }],
    "max_tokens": 400,
    "temperature": 0.0,
}
resp = brt.invoke_model(modelId=CLAUDE_MODEL, contentType="application/json", accept="application/json", body=json.dumps(body))
out = json.loads(resp["body"].read())
ans = "".join([p.get("text","") for p in out.get("content",[]) if p.get("type")=="text"])
print(ans)


#new no 3 
import faiss, numpy as np, json

# Build chunks (dict-like row access)
chunks = [
    f"Metric: {r['MetricName']}\n"
    f"Definition: {r['Definition']}\n"
    f"Derivation: {r['Derivation']}\n"
    f"SQL: {r['SQL/SourceTable']}"
    for _, r in df.iterrows()
]

# Embed function
def embed(text):
    resp = brt.invoke_model(
        modelId=EMBED_MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    return np.array(json.loads(resp["body"].read())["embedding"], dtype="float32")

# Embed corpus
X = np.vstack([embed(c) for c in chunks]).astype("float32")

# Normalize for cosine
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# FAISS index
dim = Xn.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(Xn)

# Test retrieval
query = "How is LTV calculated?"
qv = embed(query).astype("float32")
qv_n = qv / (np.linalg.norm(qv) + 1e-12)

D, I = index.search(qv_n[None, :], k=5)
sims = (Xn[I[0]] @ qv_n).tolist()  # cosine since rows are normalized

for rank, (idx, sim) in enumerate(zip(I[0], sims), start=1):
    print(f"#{rank} sim≈{sim:.2f}\n{chunks[idx]}\n---")

