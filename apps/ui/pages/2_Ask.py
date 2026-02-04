import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Ask", layout="wide")
st.title("Ask (RAG)")

docs = requests.get(f"{API_BASE}/docs", timeout=10).json()
doc_options = [d["doc_id"] for d in docs]

selected = st.multiselect("Choose guideline docs", options=doc_options)
question = st.text_area("Question", height=80, placeholder="e.g., When should antibiotics be started?")
top_k = st.slider("top_k", 1, 10, 5)

if st.button("Ask", type="primary", disabled=(len(question.strip()) < 3)):
    payload = {"question": question, "doc_ids": selected, "top_k": top_k, "mode": "rag"}
    r = requests.post(f"{API_BASE}/ask", json=payload, timeout=60)
    if r.status_code >= 400:
        st.error(r.text)
    else:
        out = r.json()
        st.subheader("Answer")
        st.write(out["answer"])
        st.subheader("Citations")
        st.write(out.get("citations", []))
        st.caption(f"request_id: {out['meta']['request_id']} | latency: {out['meta']['latency_ms']} ms")
