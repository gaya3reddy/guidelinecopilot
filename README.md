
````md
# GuidelineCopilot

Evidence-grounded RAG assistant for **public medical guidelines**  
(**PDF ingest → chunking → embeddings → retrieval → cited answers**).

Built as a portfolio project for ML/AI internship applications.

> ⚠️ Educational use only. Do not upload patient data. Not a medical device.

---

## Features

### Day 1
- Upload & ingest guideline PDFs
- Document registry (`GET /documents`)
- Duplicate upload detection (hash-based)
- Streamlit UI for ingest + viewing available docs
- FastAPI backend

### Day 2 ✅
- PDF text extraction (page-level)
- Chunking with overlap
- Embeddings (OpenAI) + persistent vector store (ChromaDB)
- RAG Q&A endpoint (`POST /ask`) returning **answer + citations**
- Ingest now returns real counts: `pages`, `chunks_indexed`

### Day 3 ✅
- Streamlit Ask page polished (citations table + expanders)
- Evidence tab (audit trail): shows retrieved snippets with `doc_id`, `page`, `chunk_id`, `score`
- Upload UX improved (shows ingest summary + stores last uploaded `doc_id` in session)

---

## Tech Stack
- **FastAPI** (API)
- **Streamlit** (UI)
- **Pydantic** (schemas)
- **pypdf** (PDF text extraction)
- **OpenAI** (chat + embeddings)
- **ChromaDB** (persistent vector store)

Planned:
- Docker / docker-compose (api + ui)
- Evaluation harness (faithfulness + citation coverage + latency)
- CI (lint + tests)

---

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

### 3) Create `.env`

Create a `.env` file in project root:

```env
APP_NAME=guidelinecopilot-api
APP_VERSION=0.1.0
API_HOST=0.0.0.0
API_PORT=8000

MODEL_PROVIDER=openai
OPENAI_API_KEY=YOUR_KEY_HERE                 ###### update with key #####
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

DATA_DIR=data
MAX_UPLOAD_MB=30
```

### 4) Run API

```bash
uvicorn apps.api.main:app --reload --port 8000
```

### 5) Run Streamlit UI

```bash
streamlit run apps/ui/Home.py
```

Open:

* API health: [http://localhost:8000/health](http://localhost:8000/health)
* API docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)
* UI: [http://localhost:8501](http://localhost:8501)

---

## Demo Flow (2 minutes)
1) Open Streamlit UI → **Upload** → ingest a public guideline PDF
2) Go to **Ask** → select the ingested `doc_id` → ask a question
3) Go to **Evidence** → view retrieved snippets + scores (traceability)

## API Overview

### `POST /ingest`

Uploads a guideline PDF, extracts text, chunks, embeds, and indexes into Chroma.

Returns:

* `doc_id`
* `pages`
* `chunks_indexed`
* `deduped` + `message` (if duplicate detected)

### `GET /documents`

Lists available docs in the current session (in-memory registry for now).

### `POST /ask`

RAG Q&A over indexed guideline chunks.

Example request:

```json
{
  "question": "What is the guideline about?",
  "doc_ids": ["doc_1234abcd"],
  "top_k": 5,
  "mode": "rag"
}
```

Response includes:

* `answer`
* `citations[]` with `doc_id`, `page`, `chunk_id`, `snippet`, `score`

---

## Notes / Current Limitations

* `/documents` registry is **in-memory** (resets on server restart). PDFs + Chroma persistence stay on disk.
* Only **public guideline PDFs** are supported (no patient data).
* Chunk quality depends on PDF text extraction quality (some PDFs contain headers/footers/license pages).

---

## Project Roadmap

* [x] Day 1: Ingest UI + doc registry + duplicate detection
* [x] Day 2: Text extraction + chunking + embeddings + retrieval + `/ask` with citations
* [x] Day 3: Ask UI polish + Evidence view (audit trail)
* [ ] Summarize endpoint + UI (grounded summaries + citations)
* [ ] Evaluation suite (latency + citation coverage + faithfulness checks)
* [ ] Dockerize (docker-compose: api + ui)
* [ ] CI (ruff + tests)

````

---

