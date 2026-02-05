
````
# GuidelineCopilot

Evidence-grounded RAG assistant for **public medical guidelines** (PDF ingest → Q&A → cited answers).
Built as a portfolio project for ML/AI internship applications.

> ⚠️ Educational use only. Do not upload patient data. Not a medical device.

## Features (Day 1)
- Upload & ingest guideline PDFs
- Document registry (`/documents`)
- Duplicate upload detection (hash-based)
- Streamlit UI for ingest + viewing available docs
- FastAPI backend

## Tech Stack
- FastAPI (API)
- Streamlit (UI)
- Pydantic (schemas)
- Docker (planned)
- Vector DB & embeddings (planned)
- Evaluation harness (planned)

## Quickstart

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
````

### 2) Install dependencies

```bash
pip install -U pip
pip install -e ".[dev]"
```

### 3) Run API

```bash
uvicorn apps.api.main:app --reload --port 8000
```

### 4) Run Streamlit UI

```bash
streamlit run apps/ui/Home.py
```

Open:

* API health: [http://localhost:8000/health](http://localhost:8000/health)
* API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* UI: [http://localhost:8501](http://localhost:8501)

## Project Roadmap

* [ ] Day 2–7: Text extraction + chunking + embeddings + retrieval
* [ ] Add Ask page (RAG) with citations
* [ ] Add evaluation suite (faithfulness + citation coverage + latency)
* [ ] Dockerize (docker-compose: api + ui)
* [ ] CI (lint + tests)


````
