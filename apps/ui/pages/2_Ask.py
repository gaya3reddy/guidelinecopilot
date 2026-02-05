import os
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Ask", layout="wide")
st.title("Ask (RAG)")

# ---- Load documents ----
docs = []
try:
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    r.raise_for_status()
    data = r.json()
    docs = data.get("items", [])
except Exception as e:
    st.error(f"Could not load documents: {e}")

doc_options = [d["doc_id"] for d in docs] if docs else []

selected = st.multiselect("Choose guideline docs", options=doc_options)
mode = st.radio("Mode", options=["rag", "no_rag"], horizontal=True)
top_k = st.slider("top_k", 1, 10, 5)
question = st.text_area("Question", height=90, placeholder="e.g., What does this guideline recommend?")

if st.button("Ask", type="primary", disabled=(len(question.strip()) < 3)):
    payload = {"question": question.strip(), "doc_ids": selected, "top_k": top_k, "mode": mode}
    r = requests.post(f"{API_BASE}/ask", json=payload, timeout=90)

    if not r.ok:
        st.error(f"API error {r.status_code}")
        st.code(r.text[:2000])
    else:
        out = r.json()

        st.subheader("Answer")
        st.write(out["answer"])

        st.subheader("Citations")
        cits = out.get("citations", [])

        if not cits:
            st.info("No citations returned.")
        else:
            # Table summary
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "doc_id": c["doc_id"],
                            "page": c["page"],
                            "chunk_id": c["chunk_id"],
                            "score": round(c["score"], 3),
                        }
                        for c in cits
                    ]
                )
            )
            # Expanders with snippets
            for i, c in enumerate(cits, start=1):
                with st.expander(f"[{i}] {c['doc_id']} • page {c['page']} • score {c['score']:.3f}"):
                    st.write(c["snippet"])

        meta = out.get("meta", {})
        st.caption(f"request_id: {meta.get('request_id')} | latency: {meta.get('latency_ms')} ms | model: {meta.get('model')}")
