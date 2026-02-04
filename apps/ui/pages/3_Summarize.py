import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Summarize", layout="wide")
st.title("Summarize")

docs = requests.get(f"{API_BASE}/docs", timeout=10).json()
doc_options = [d["doc_id"] for d in docs]
selected = st.multiselect("Choose guideline docs", options=doc_options)

style = st.radio("Style", ["tldr", "key_steps", "contraindications", "eligibility"], horizontal=True)

if st.button("Summarize", type="primary"):
    payload = {"doc_ids": selected, "style": style}
    r = requests.post(f"{API_BASE}/summarize", json=payload, timeout=60)
    if r.status_code >= 400:
        st.error(r.text)
    else:
        out = r.json()
        st.subheader("Summary")
        st.write(out["summary"])
        st.caption(f"request_id: {out['meta']['request_id']} | latency: {out['meta']['latency_ms']} ms")
