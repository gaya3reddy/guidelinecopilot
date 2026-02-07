import streamlit as st
import pandas as pd

st.set_page_config(page_title="Evidence", layout="wide")
st.title("Evidence")

last = st.session_state.get("last_ask")
payload = st.session_state.get("last_ask_payload")

if not last:
    st.info("No recent Ask result found. Go to **Ask** tab, ask a question, then come back here.")
    st.stop()

st.subheader("Last query")
if payload:
    st.code(payload, language="json")

cits = last.get("citations", [])
if not cits:
    st.warning("No citations available in last result.")
    st.stop()

st.subheader("Retrieved evidence (citations)")
df = pd.DataFrame(
    [
        {
            "doc_id": c["doc_id"],
            "page": c["page"],
            "chunk_id": c["chunk_id"],
            "score": round(c["score"], 3),
            "snippet_preview": (c["snippet"][:120] + "…") if len(c["snippet"]) > 120 else c["snippet"],
        }
        for c in cits
    ]
)
st.dataframe(df, use_container_width=True)

st.markdown("---")
st.subheader("Full snippets")
for i, c in enumerate(cits, start=1):
    with st.expander(f"[{i}] {c['doc_id']} • page {c['page']} • chunk {c['chunk_id']} • score {c['score']:.3f}"):
        st.write(c["snippet"])
