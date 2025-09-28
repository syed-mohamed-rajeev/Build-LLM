# app_fullkb_bedrock_identifiers_clean.py
# Enterprise Risk Analytics Desk - TraceGenie
# Full-KB-to-LLM (Bedrock Inference Profile + PII pre-check)
# - Metric questions: answer + (optional) expanders + lineage
# - Identifier-only questions (Interface ID / IAG Roles): answer only (no expanders, no lineage)
# - White UI, blue assistant bubble, dark-node lineage on white canvas

from __future__ import annotations
import json, uuid, re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set, Optional

import boto3
import streamlit as st

# =========================
# Page config + global styles
# =========================
st.set_page_config(page_title="Enterprise Risk Desk", page_icon="🏢", layout="wide")

st.markdown("""
<style>
/* Full-app white */
html, body, .stApp, .main, .block-container { background:#ffffff !important; }
.block-container { padding-top:.8rem; max-width:1400px; }

/* White top bars */
header, [data-testid="stHeader"], [data-testid="stToolbar"] { background:#ffffff !important; }

/* Chat input: white bg, darker text */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] div[contenteditable="true"],
[data-testid="stChatInput"] input[type="text"],
[data-testid="stChatInput"] {
  background:#ffffff !important;
  color:#0b122a !important;
  border:1px solid rgba(15,23,42,.25) !important;
  border-radius:10px !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#475569 !important; opacity:1 !important; }
.stChatInputContainer, [data-testid="stBottomBlockContainer"] { background:#ffffff !important; }

/* Bubbles */
.bubble { border-radius:16px; padding:12px 14px; margin:10px 0; border:1px solid rgba(148,163,184,.25); }
.user { background:#f8fafc; color:#0f172a; }
.assistant {
  background:linear-gradient(135deg,#1e40af 0%,#2563eb 100%);
  border-color:rgba(37,99,235,.55); color:#ffffff;
}
.small { opacity:.88; font-size:.92rem; }
.message-wrap { margin-bottom:.4rem; }
.analyzing { background:rgba(234,179,8,.10); border:1px dashed rgba(234,179,8,.45); color:#1f2937; }

/* Expander headers + content darker */
[data-testid="stExpander"] summary, details > summary {
  color:#0f172a !important; font-weight:700 !important; font-size:1.02rem !important;
}
[data-testid="stExpander"] div[role="region"], details > div { color:#0f172a !important; }
[data-testid="stExpander"] { border:1px solid rgba(15,23,42,.12); border-radius:12px; background:#ffffff; }

/* Data Lineage header */
h3, .stMarkdown h3 { color:#0f172a !important; font-weight:800 !important; }

/* Graphviz */
iframe[title="graphviz-chart"] { background:#ffffff !important; }

/* HR */
hr.soft { border:none; border-top:1px solid rgba(148,163,184,.25); margin:.75rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* --- Analyze bubble light blue --- */
.bubble.analyzing {
  background: #e0f2fe !important;  /* light blue */
  border: 1px dashed #38bdf8 !important; /* cyan border */
  color: #0c4a6e !important; /* dark teal text */
}


/* --- Darker Data Lineage heading --- */
h3 {
  color: #0f172a !important;
  font-weight: 700 !important;
}

/* --- Softer expander header text --- */
.streamlit-expanderHeader {
  background-color: #f8fafc !important; /* light header bg */
  color: #334155 !important;           /* dark gray text (softer than black) */
  font-weight: 600 !important;
  font-size: 1rem !important;
}

/* --- Expander content text --- */
.streamlit-expanderContent {
  color: #1e293b !important;   /* dark slate for content */
  font-size: .95rem !important;
}

/* --- Darker user input text --- */
.stTextInput > div > input,
.stTextArea > div > textarea,
.stChatInput input {
  color: #0f172a !important;  /* dark slate text */
  font-weight: 500 !important;
  background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Config (from secrets)
# =========================
try:
    BEDROCK_REGION = st.secrets["bedrock"]["region"]
    INFERENCE_PROFILE_ARN = st.secrets["bedrock"]["inference_profile_arn"]
    GUARDRAIL_IDENTIFIER = st.secrets["bedrock"]["guardrail_identifier"]
    GUARDRAIL_VERSION = str(st.secrets["bedrock"]["guardrail_version"])
except KeyError:
    st.error("Missing Bedrock config in .streamlit/secrets.toml under [bedrock]: region, inference_profile_arn, guardrail_identifier, guardrail_version")
    st.stop()

DEFAULT_KB_PATH = st.secrets.get("kb", {}).get("path", "synthetic_mortgage_kb_50_with_new_columns.jsonl")

# Tunables
TEMPERATURE       = 0.2
MAX_TOKENS        = 900
MAX_CTX_CHARS     = 18000
PER_ROW_DDL_CHARS = 350
MIN_SCORE         = 0.22
TOP_ROWS_FOR_LINEAGE = 3

# =========================
# Data model & cache
# =========================
@dataclass
class KBRow:
    id: str
    text: str
    metadata: Dict[str, Any]

@st.cache_resource(show_spinner=False)
def get_bedrock_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)

def bedrock_client():
    return get_bedrock_client(BEDROCK_REGION)

# Loader merges top-level extras (interface_id, IAG_roles, etc.) into metadata
@st.cache_data(show_spinner=False)
def load_kb_jsonl(path: str) -> List[KBRow]:
    rows: List[KBRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)

            kb_id   = obj.get("id") or str(uuid.uuid4())
            text    = obj.get("text") or ""
            meta_in = obj.get("metadata") or {}

            core_keys = {"id", "text", "metadata"}
            extras = {k: v for k, v in obj.items() if k not in core_keys}

            merged_meta = dict(meta_in)
            merged_meta.update(extras)  # brings interface_id, IAG_roles into metadata

            rows.append(KBRow(id=kb_id, text=text, metadata=merged_meta))
    return rows

# =========================
# System prompt (include-only-what-you-asked for identifiers)
# =========================
SYSTEM_MSG = """
You are a senior data scientist for mortgage risk (single-family & multi-family).

USE ONLY THE PROVIDED KB CONTEXT.
Do NOT invent tables/fields/views/identifiers not present in the context.
Do NOT output SQL/DDL/code; explain derivations in words.
If a view DDL mentions multiple metrics, explain ONLY the metric(s) the user asked for.
Cite kb_id inline like [kb_id=...].
If details are missing, say “Not available in KB.”

Identifiers policy:
- If the user asks for identifiers, include ONLY the identifiers explicitly requested.
- If the user asks for interface id: include `interface_id` and OMIT `IAG_roles`.
- If the user asks for IAG role(s): include `IAG_roles` and OMIT `interface_id`.
- If the user asks for both, include both.
- Do NOT say “Not available in KB” for identifiers the user did NOT ask for; simply omit them.
- Do not fabricate identifiers.

!!! OUTPUT FORMAT (STRICT JSON) !!!
Return ONLY a single JSON object with these EXACT keys:

{
  "answer": "Concise business-facing answer (markdown allowed). No numbering. No 'Executive Summary'.",
  "definitions_derivation": "Definition, formula in words, units/scale, interpretation (markdown).",
  "view_definition_insight": "From pg_views_definition_aggregated: how THIS metric is derived in that view; ignore other metrics; if not available, say 'Not available in KB.' (markdown)"
}

No extra keys, no commentary before/after the JSON.
"""

# =========================
# Build Full-KB context
# =========================
def _short(v: Any, n: int = 600) -> str:
    s = ("" if v is None else str(v)).strip()
    return (s[:n] + "…") if len(s) > n else s

def infer_segment_from_id(kid: str) -> str:
    k = (kid or "").lower()
    if k.startswith("sf_"): return "Single-Family"
    if k.startswith("mf_"): return "Multi-Family"
    return "Unknown"

def format_kb_block(r: KBRow, ddl_chars: int) -> str:
    m = r.metadata or {}
    seg = infer_segment_from_id(r.id)
    return (
        f"[kb_id={r.id}] (segment={seg})\n"
        f"- Text: {_short(r.text, 500)}\n"
        f"- Logical Definition: {_short(m.get('logical_definition',''), 500)}\n"
        f"- Source System: {m.get('source_system','')}\n"
        f"- Table: {m.get('loan_level_table','')} | Field: {m.get('loan_level_field_name','')}\n"
        f"- PG View: {m.get('pg_views_loan_level','')}\n"
        f"- View DDL (aggregated): {_short(m.get('pg_views_definition_aggregated',''), ddl_chars)}\n"
        f"- Tableau Metric: {m.get('tableau_metric_name','')} | Report: {m.get('tableau_report_name','')}\n"
        f"- Interface ID: {m.get('interface_id','')}\n"
        f"- IAG Roles: {m.get('IAG_roles','')}\n"
    )

@st.cache_data(show_spinner=False, max_entries=8)
def build_full_kb_context(kb: List[KBRow], max_chars: int, per_row_dl_chars: int) -> str:
    header = [
        "KB CONTRACT:",
        "- Use only the facts below.",
        "- Cite [kb_id=...] for any statement tied to a row.",
        "- Do NOT output SQL/DDL; explain derivations in words.",
        "- If pg_views_definition_aggregated contains other metrics, IGNORE them.",
        ""
    ]
    blocks: List[str] = ["\n".join(header)]
    used = len(blocks[0])
    for r in kb:
        block = format_kb_block(r, ddl_chars=per_row_dl_chars)
        if used + len(block) > max_chars: break
        blocks.append(block); used += len(block)
    return "\n".join(blocks)

# =========================
# Lightweight relevance (for lineage)
# =========================
METRIC_HINTS = [
    "ltv","loan-to-value","loan to value",
    "dti","debt-to-income","debt to income",
    "dscr","pd","probability of default","lgd","ead",
    "cpr","cdr","npl","wac","wam","occupancy","cap","icr","yield","duration","delinquency"
]
SEG_HINTS = {
    "Single-Family": ["single family","single-family","sf"],
    "Multi-Family":  ["multi family","multifamily","multi-family","mf"]
}

def infer_seg_from_text(text: str) -> Optional[str]:
    t = (text or "").lower()
    for seg, keys in SEG_HINTS.items():
        if any(k in t for k in keys): return seg
    return None

def score_row(user_text: str, row: KBRow, seg_pref: Optional[str]) -> float:
    t = (user_text or "").lower()
    m = row.metadata or {}
    text_blobs = [
        row.text, m.get("logical_definition",""), m.get("loan_level_table",""),
        m.get("loan_level_field_name",""), m.get("pg_views_loan_level",""),
        m.get("pg_views_definition_aggregated",""), m.get("tableau_metric_name",""),
        m.get("tableau_report_name",""),
        m.get("interface_id",""), m.get("IAG_roles","")
    ]
    blob = " ".join([b for b in text_blobs if b]).lower()
    hits = sum(1 for kw in METRIC_HINTS if kw in t and kw in blob)
    tokens = set(re.findall(r"[a-z0-9]+", t))
    hits += sum(0.2 for tok in tokens if tok and tok in blob)
    if seg_pref == "Single-Family" and row.id.lower().startswith("sf_"): hits += 1.0
    if seg_pref == "Multi-Family" and row.id.lower().startswith("mf_"): hits += 1.0
    tm = (m.get("tableau_metric_name","") or "").lower()
    if tm and tm in t: hits += 1.0
    return float(hits)

def pick_relevant_rows(user_text: str, kb: List[KBRow], top_n: int) -> List[KBRow]:
    seg_pref = infer_seg_from_text(user_text or "")
    if not kb: return []
    scored = [(r, float(score_row(user_text, r, seg_pref))) for r in kb]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for (r, s) in scored[:max(int(top_n or 0), 1)] if s > MIN_SCORE]

# =========================
# PII pre-check (strict, client-side)
# =========================
PII_PATTERNS = [
    re.compile(r"\bssn\b", re.I),
    re.compile(r"\bsocial\s*security\s*number\b", re.I),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{9}\b"),
]
def looks_like_pii_prompt(txt: str) -> bool:
    t = txt or ""
    return any(p.search(t) for p in PII_PATTERNS)

# =========================
# Lineage (Graphviz) — dark nodes on white
# =========================
NODE_STYLE = {
    "source":  {"fill":"#0b132b", "border":"#3b82f6", "shape":"folder"},
    "table":   {"fill":"#102a43", "border":"#22c55e", "shape":"component"},
    "pgview":  {"fill":"#1f2937", "border":"#f59e0b", "shape":"tab"},
    "field":   {"fill":"#2b1d2e", "border":"#a855f7", "shape":"note"},
    "metric":  {"fill":"#1e293b", "border":"#eab308", "shape":"parallelogram"},
    "report":  {"fill":"#3a1726", "border":"#ec4899", "shape":"box3d"},
    "unknown": {"fill":"#334155", "border":"#64748b", "shape":"box"},
}
def _node_type(name: str) -> str:
    n = (name or "").lower()
    if re.search(r"(datamart|edl|source|system)", n): return "source"
    if re.search(r"(^dim_|^fact_|_metrics$|table|loan)", n): return "table"
    if any(k in n for k in ["pg_view","pg views","pg_views","vw_"]): return "pgview"
    if re.search(r"(field|column|ratio|score|rate|amount|dti|ltv|dscr|lgd|pd|ead|cpr|wam|wac|npl|occupancy|cap|icr|yield|duration|delinquency)", n): return "field"
    if "tableau_metric" in n or " metric" in n: return "metric"
    if any(k in n for k in ["tableau_report","dashboard","report"]): return "report"
    return "unknown"

def _node_decl(name: str) -> str:
    t = _node_type(name); sty = NODE_STYLE[t]
    s = (name or "").replace('"','\\"')
    return f"\"{s}\" [shape={sty['shape']}, style=\"rounded,filled\", fillcolor=\"{sty['fill']}\", color=\"{sty['border']}\", fontcolor=\"#ffffff\"];"

def lineage_edges_from_rows(rows: List[KBRow]) -> Tuple[List[Tuple[str,str,str]], List[str]]:
    edges, nodes = [], set()
    for r in rows:
        m = r.metadata or {}
        src  = m.get("source_system","")
        tbl  = m.get("loan_level_table","")
        fld  = m.get("loan_level_field_name","")
        pgv  = m.get("pg_views_loan_level","")
        tmet = m.get("tableau_metric_name","")
        trep = m.get("tableau_report_name","")
        def add(a,b,label):
            if a and b:
                edges.append((a,b,label)); nodes.add(a); nodes.add(b)
        add(src, tbl, "source → table")
        add(tbl, pgv, "table → view")
        add(tbl, fld, "table → field")
        add(fld, tmet, "field → metric")
        add(tmet, trep, "metric → report")
    uniq, seen = [], set()
    for e in edges:
        if e not in seen:
            uniq.append(e); seen.add(e)
    return uniq, list(nodes)

def _metric_keywords_from_question(user_text: str) -> List[str]:
    t = (user_text or "").lower()
    pairs = [
        (["ltv","loan-to-value","loan to value"], "ltv"),
        (["dti","debt-to-income","debt to income"], "dti"),
        (["dscr"], "dscr"),
        (["pd","probability of default"], "pd"),
        (["lgd"], "lgd"),
        (["ead"], "ead"),
        (["cpr"], "cpr"),
        (["cdr"], "cdr"),
        (["npl","non-performing"], "npl"),
    ]
    keys = []
    for variants, tag in pairs:
        if any(v in t for v in variants):
            keys.extend(variants + [tag])
    return list(dict.fromkeys(keys))

def filter_edges_for_metric(
    edges: List[Tuple[str, str, str]],
    keywords: List[str],
    max_nodes: int = 30
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    if not edges or not keywords:
        return [], []
    kw = [k.lower() for k in keywords]
    keep: Set[Tuple[str, str, str]] = set()
    nodes: Set[str] = set()
    for (a, b, label) in edges:
        al, bl = a.lower(), b.lower()
        if any(k in al for k in kw) or any(k in bl for k in kw):
            keep.add((a, b, label)); nodes.add(a); nodes.add(b)
    if not keep:
        return [], []
    for (a, b, label) in edges:
        if a in nodes or b in nodes:
            keep.add((a, b, label)); nodes.add(a); nodes.add(b)
        if len(nodes) >= max_nodes:
            break
    return list(keep), list(nodes)

def graphviz_from_edges(edges: List[Tuple[str,str,str]], nodes: List[str]) -> str:
    lines = [
        'digraph G {',
        'rankdir=LR;',
        'graph [pad="0.2", nodesep="0.45", ranksep="0.65", bgcolor="#ffffff"];',
        'node  [fontname="Inter", fontsize=12, color="#93c5fd", fontcolor="#ffffff"];',
        'edge  [color="#3b82f6", arrowsize=0.9, penwidth=1.3, fontname="Inter", fontsize=10, fontcolor="#1f2937"];'
    ]
    for n in nodes: lines.append(_node_decl(n))
    for a,b,label in edges:
        a_s, b_s = a.replace('"','\\"'), b.replace('"','\\"')
        lines.append(f"\"{a_s}\" -> \"{b_s}\" [label=\"{label}\"];")
    lines.append('}')
    return "\n".join(lines)

# =========================
# Bedrock (non-streaming)
# =========================
def bedrock_chat(system: str, user: str, temperature: float, max_tokens: int) -> Tuple[str, dict, dict]:
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = bedrock_client().invoke_model(
        modelId=INFERENCE_PROFILE_ARN,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload),
        guardrailIdentifier=GUARDRAIL_IDENTIFIER,
        guardrailVersion=GUARDRAIL_VERSION,
    )
    body = json.loads(resp["body"].read())
    headers = resp.get("ResponseMetadata", {}).get("HTTPHeaders", {}) or {}
    parts = body.get("content", [])
    out = []
    for p in parts:
        if p.get("type") == "text" and "text" in p: out.append(p["text"])
        elif p.get("text"): out.append(p["text"])
    answer_text = "".join(out)
    return answer_text, body, headers

def detect_guardrail_block(answer_text: str, body: dict, headers: dict) -> bool:
    return not (answer_text or "").strip()

# =========================
# JSON extraction
# =========================
def extract_json_payload(text: str) -> Dict[str, str]:
    if not text:
        return {"answer": "", "definitions_derivation": "", "view_definition_insight": ""}
    try:
        obj = json.loads(text)
        return {
            "answer": obj.get("answer",""),
            "definitions_derivation": obj.get("definitions_derivation",""),
            "view_definition_insight": obj.get("view_definition_insight",""),
        }
    except Exception:
        pass
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            return {
                "answer": obj.get("answer",""),
                "definitions_derivation": obj.get("definitions_derivation",""),
                "view_definition_insight": obj.get("view_definition_insight",""),
            }
        except Exception:
            pass
    return {"answer": text.strip(), "definitions_derivation": "", "view_definition_insight": ""}

# =========================
# Load KB
# =========================
try:
    KB = load_kb_jsonl(DEFAULT_KB_PATH)
except Exception as e:
    st.error(f"Could not load KB at '{DEFAULT_KB_PATH}': {e}")
    st.stop()

# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_user_text" not in st.session_state:
    st.session_state["last_user_text"] = ""

# ========== Header / Branding ==========
st.markdown("""
<h1 style="
  font-size:2.3rem;
  font-weight:700;
  background:linear-gradient(90deg,#2563eb,#60a5fa,#2563eb);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  animation: shine 4s infinite;
  background-size:200% auto;
  margin-bottom:0.2rem;
">
 Enterprise Risk Analytics Desk - TraceGenie
</h1>
<style>
@keyframes shine { to { background-position:200% center; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    font-size:1.05rem;
    font-weight:600;
    color:#1d4ed8;
    margin-top:-0.35rem;
    margin-bottom:0.8rem;
">
Know your numbers, trust your metrics – with TraceGenie.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    padding:16px 20px;
    border-radius:14px;
    color:#f1f5f9;
    font-size:1.02rem;
    line-height:1.5;
    font-weight:400;
    box-shadow:0 3px 10px rgba(0,0,0,0.15);
    margin-bottom:1rem;
">
Enterprise Risk Analytics Desk provides clear business-friendly insights into single-family and multi-family metrics supported transparent visual lineage for traceable, trusted decision-making.
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# =========================
# Identifier helpers (guide LLM and toggle lineage/expanders)
# =========================
# We recognize "interface id", "interface identifier", and also looser "interface for ..."
INTERFACE_PAT = re.compile(
    r"\b(interface(?:[_\s-]?id|[\s-]*identifier)|interface\s+for\b)", re.I
)

# Treat "iag roles", "iag role", and also **"iag for ..."** as IAG-roles intent
IAG_PAT = re.compile(
    r"\b(iag(?:[_\s-]*roles?)?\b(?:\s+for\b)?)", re.I
)

def identifier_directive(user_text: str) -> Tuple[str, bool]:
    """
    Returns (directive_text, identifier_only_flag).
    identifier_only_flag=True => skip expanders and lineage for this turn.
    """
    t = (user_text or "").strip().lower()

    # Primary regex matches
    want_iface = bool(INTERFACE_PAT.search(t))
    want_iag   = bool(IAG_PAT.search(t))

    # Additional heuristics to catch natural phrasing:
    # - "iag for ltv" (no 'role' word) -> treat as IAG roles
    if ("iag" in t and "for" in t) or ("iag:" in t) or ("iag -" in t):
        want_iag = True

    # - "interface for ltv" (no 'id' word) -> treat as interface id
    if ("interface" in t and "for" in t) and not want_iface:
        want_iface = True

    # - explicit single keywords (short prompts): "iag ltv", "interface ltv"
    if t.startswith("iag ") or t.endswith(" iag"):
        want_iag = True
    if t.startswith("interface ") or t.endswith(" interface"):
        want_iface = True

    # Resolve directive
    if want_iface and not want_iag:
        return ("For identifiers, include ONLY `interface_id`. Do NOT mention `IAG_roles`.", True)
    if want_iag and not want_iface:
        return ("For identifiers, include ONLY `IAG_roles`. Do NOT mention `interface_id`.", True)
    if want_iface and want_iag:
        return ("For identifiers, include BOTH `interface_id` and `IAG_roles` when present.", True)
    return ("", False)
# ========== Render chat history ==========
def render_message(msg: Dict[str, Any]):
    if msg["role"] == "user":
        st.markdown(f"<div class='bubble user'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bubble assistant'>{msg['answer'] if msg['answer'] else '_No content returned._'}</div>", unsafe_allow_html=True)
        if not msg.get("guarded", False):
            if msg.get("defs"):
                with st.expander("Definitions & Derivation", expanded=False):
                    st.markdown(msg["defs"])
            if msg.get("view"):
                with st.expander("View Definition Insight", expanded=False):
                    st.markdown(msg["view"])
            if msg.get("dot"):
                st.markdown("### 🧭 Data Lineage")
                st.graphviz_chart(msg["dot"], use_container_width=True)
        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

for m in st.session_state["messages"]:
    render_message(m)

# =========================
# Chat input
# =========================
user_text = st.chat_input("Ask about a metric (e.g., 'Explain single-family LTV with lineage' or 'IAG role for LTV')…")

if user_text:
    user_msg = {"role": "user", "content": user_text}
    st.session_state["messages"].append(user_msg)
    render_message(user_msg)
    st.session_state["last_user_text"] = user_text

    # PII pre-check (short-circuit)
    if looks_like_pii_prompt(user_text):
        guard_msg = "I can’t provide or process sensitive PII such as Social Security Numbers (SSNs). Please rephrase your question without PII."
        asst = {"role":"assistant","answer":guard_msg,"defs":"","view":"","dot":None,"guarded":True}
        st.session_state["messages"].append(asst)
        render_message(asst)
        st.stop()

    # Subtle analyzing status
    status = st.empty()
    status.markdown("<div class='bubble analyzing small'>🔎 Analyzing KB…</div>", unsafe_allow_html=True)

    # Build full-KB context
    headered_kb = build_full_kb_context(KB, MAX_CTX_CHARS, PER_ROW_DDL_CHARS)

    # Identifier directive + skip (expanders & lineage) flag
    dir_txt, identifier_only = identifier_directive(user_text)

    user_payload = (
        f"CONTEXT (ENTIRE KB):\n{headered_kb}\n\n"
        f"QUESTION:\n{user_text}\n\n"
        + (f"ADDITIONAL INSTRUCTIONS:\n{dir_txt}\n\n" if dir_txt else "")
        + "Return ONLY the JSON object as specified."
    )

    # LLM call
    try:
        raw_text, raw_body, headers = bedrock_chat(SYSTEM_MSG, user_payload, TEMPERATURE, MAX_TOKENS)
    except Exception as e:
        raw_text, raw_body, headers = f'{{"answer":"[Bedrock error: {e}]","definitions_derivation":"","view_definition_insight":""}}', {}, {}
    status.empty()

    # Guardrail block
    if detect_guardrail_block(raw_text, raw_body, headers):
        asst = {"role":"assistant","answer":"Response blocked by Guardrails.","defs":"","view":"","dot":None,"guarded":True}
        st.session_state["messages"].append(asst)
        render_message(asst)
        st.stop()

    # Parse JSON payload
    payload = extract_json_payload(raw_text)
    ans  = (payload.get("answer") or "").strip()
    # For identifier-only: suppress expanders
    defs = "" if identifier_only else (payload.get("definitions_derivation") or "").strip()
    view = "" if identifier_only else (payload.get("view_definition_insight") or "").strip()

    # Build lineage only if NOT an identifier-only query
    dot = None
    if not identifier_only:
        metric_keys = _metric_keywords_from_question(user_text)
        if metric_keys:
            rel_rows = pick_relevant_rows(user_text, KB, TOP_ROWS_FOR_LINEAGE)
            if rel_rows:
                edges, nodes = lineage_edges_from_rows(rel_rows)
                kept_edges, kept_nodes = filter_edges_for_metric(edges, metric_keys, max_nodes=30)
                if kept_edges and kept_nodes:
                    dot = graphviz_from_edges(kept_edges, kept_nodes)

    # Store assistant message
    asst = {"role":"assistant","answer":ans, "defs":defs, "view":view, "dot":dot, "guarded":False}
    st.session_state["messages"].append(asst)
    render_message(asst)