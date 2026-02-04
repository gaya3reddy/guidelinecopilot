import os
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Upload", layout="wide")
st.title("Upload & Ingest")

st.warning("Public guideline PDFs only. No patient data. Educational use.")

uploaded = st.file_uploader("Upload a guideline PDF", type=["pdf"])
col1, col2, col3 = st.columns(3)
title = col1.text_input("Title (optional)")
source = col2.text_input("Source (optional)")
category = col3.text_input("Category (optional)")
doc_id = st.text_input("doc_id (optional, leave empty to auto-generate)")

if st.button("Ingest", type="primary", disabled=(uploaded is None)):
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
    data = {"doc_id": doc_id, "title": title, "source": source, "category": category}
    r = requests.post(f"{API_BASE}/ingest", files=files, data=data, timeout=60)
    if r.status_code >= 400:
        st.error(r.text)
    else:
        st.success("Ingested ✅ (stub indexing for Day-1)")
        st.json(r.json())

st.markdown("---")
st.subheader("Available docs")

try:
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    if not r.ok:
        st.error(f"API error {r.status_code}")
        st.code(r.text[:1500])
    else:
        data = r.json()
        items = data.get("items", [])
        st.dataframe(pd.DataFrame(items))
        if not items:
            st.info("No documents ingested yet.")
        else:
            st.json(items)
        # st.json(data["items"])
        # st.json(r.json())
except Exception as e:
    st.error(f"Could not load available docs: {e}")
