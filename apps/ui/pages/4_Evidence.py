import streamlit as st
import pandas as pd

st.set_page_config(page_title="Evidence", layout="wide")
st.title("Evidence")

# Pull both possibilities from session state
last_ask = st.session_state.get("last_ask")
ask_payload = st.session_state.get("last_ask_payload")

last_sum = st.session_state.get("last_summary")
sum_payload = st.session_state.get("last_summary_payload")

# Decide what to show
options = []
if last_ask:
    options.append("Ask")
if last_sum:
    options.append("Summarize")

if not options:
    st.info("No recent Ask/Summarize result found. Run **Ask** or **Summarize**, then come back here.")
    st.stop()

mode = st.radio("Show evidence for", options=options, horizontal=True)

if mode == "Ask":
    last = last_ask
    payload = ask_payload
    st.subheader("Last Ask payload")
elif mode == "Summarize":
    last = last_sum
    payload = sum_payload
    st.subheader("Last Summarize payload")

if payload:
    st.code(payload, language="json")

# Extract citations
cits = (last or {}).get("citations", [])
if not cits:
    st.warning("No citations available in this result.")
    st.stop()

st.subheader("Retrieved evidence (citations)")
df = pd.DataFrame(
    [
        {
            "doc_id": c.get("doc_id"),
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
            "score": round(float(c.get("score", 0.0)), 3),
            "snippet_preview": (
                (c.get("snippet", "")[:140] + "…")
                if len(c.get("snippet", "")) > 140
                else c.get("snippet", "")
            ),
        }
        for c in cits
    ]
)
st.dataframe(df, use_container_width=True)

st.markdown("---")
st.subheader("Full snippets")
for i, c in enumerate(cits, start=1):
    doc_id = c.get("doc_id", "")
    page = c.get("page", "")
    chunk_id = c.get("chunk_id", "")
    score = float(c.get("score", 0.0))
    with st.expander(f"[{i}] {doc_id} • page {page} • chunk {chunk_id} • score {score:.3f}"):
        st.write(c.get("snippet", ""))
