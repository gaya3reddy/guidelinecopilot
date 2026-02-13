import os
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Ask", layout="wide")
st.title("Ask (RAG)")

# Sidebar: API health
with st.sidebar:
    st.markdown("### API")
    try:
        hr = requests.get(f"{API_BASE}/health", timeout=5)
        st.success("API connected ✅" if hr.ok else "API error ❌")
        if hr.ok:
            st.caption(hr.json())
    except Exception:
        st.error("API not reachable")

#  Load documents for selection
docs = []
try:
    r = requests.get(f"{API_BASE}/documents", timeout=10)
    r.raise_for_status()
    docs = r.json().get("items", [])
except Exception as e:
    st.error(f"Could not load documents: {e}")

doc_options = [d["doc_id"] for d in docs] if docs else []

last_doc = st.session_state.get("last_ingested_doc_id")
if last_doc and last_doc in doc_options:
    if st.button("Use last uploaded doc"):
        st.session_state["prefill_doc_ids"] = [last_doc]


colA, colB = st.columns([2, 1])

with colA:
    # selected = st.multiselect("Choose guideline docs", options=doc_options)
    default_selected = st.session_state.pop("prefill_doc_ids", [])
    selected = st.multiselect(
        "Choose guideline docs", options=doc_options, default=default_selected
    )
    question = st.text_area(
        "Question",
        height=90,
        placeholder="e.g., What is this guideline about? What does it recommend?",
    )

with colB:
    mode = st.radio("Mode", options=["rag", "no_rag"], horizontal=True)
    top_k = st.slider("top_k", 1, 10, 5)
    run = st.button(
        "Ask",
        type="primary",
        use_container_width=True,
        disabled=(len(question.strip()) < 3),
    )

if run:
    payload = {
        "question": question.strip(),
        "doc_ids": selected,
        "top_k": top_k,
        "mode": mode,
    }
    r = requests.post(f"{API_BASE}/ask", json=payload, timeout=90)

    if not r.ok:
        st.error(f"API error {r.status_code}")
        st.code(r.text[:2000])
    else:
        out = r.json()
        # store for Evidence page
        st.session_state["last_ask"] = out
        st.session_state["last_ask_payload"] = payload

        st.subheader("Answer")
        st.write(out.get("answer", ""))

        cits = out.get("citations", [])
        st.subheader("Citations")
        if not cits:
            st.info("No citations returned.")
        else:
            # Summary table
            df = pd.DataFrame(
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
            # st.dataframe(df, use_container_width=True)
            st.dataframe(df, width="stretch")

            # Detailed evidence expanders
            for i, c in enumerate(cits, start=1):
                with st.expander(
                    f"[{i}] {c['doc_id']} • page {c['page']} • score {c['score']:.3f}"
                ):
                    st.write(c["snippet"])

        meta = out.get("meta", {})
        st.caption(
            f"request_id: {meta.get('request_id')} | latency: {meta.get('latency_ms')} ms | model: {meta.get('model')}"
        )

        st.info(
            "Tip: Open the **Evidence** tab to view retrieved snippets as an audit trail."
        )
