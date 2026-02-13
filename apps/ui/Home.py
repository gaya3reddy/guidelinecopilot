import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="GuidelineCopilot", layout="wide")
st.title("GuidelineCopilot (ClinAssist RAG)")
st.caption(
    "Evidence-grounded Q&A + summarization over public guideline PDFs (educational use)."
)

with st.sidebar:
    st.subheader("API")
    st.write(f"Base: `{API_BASE}`")

# Health check
try:
    r = requests.get(f"{API_BASE}/health", timeout=3)
    r.raise_for_status()
    data = r.json()
    st.success(f"API connected ✅  ({data.get('service')} v{data.get('version')})")
except Exception:
    st.error("API not reachable ❌")
    st.info("Start API: `uvicorn apps.api.main:app --reload`")
    st.stop()

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/1_Upload.py", label="📄 Upload & Ingest", icon="📄")
with col2:
    st.page_link("pages/2_Ask.py", label="💬 Ask (RAG)", icon="💬")
with col3:
    st.page_link("pages/3_Summarize.py", label="🧾 Summarize", icon="🧾")
