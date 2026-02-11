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

### Day 2
- PDF text extraction (page-level)
- Chunking with overlap
- Embeddings (OpenAI) + persistent vector store (ChromaDB)
- RAG Q&A endpoint (`POST /ask`) returning **answer + citations**
- Ingest returns real counts: `pages`, `chunks_indexed`

### Day 3
- Streamlit Ask page polished (citations table + expanders)
- Evidence tab (audit trail): retrieved snippets with `doc_id`, `page`, `chunk_id`, `score`
- Upload UX improved (shows ingest summary + stores last uploaded `doc_id` in session)

### Day 4
- Added `/summarize` endpoint: produces *grounded* summaries using retrieved chunks + citations
- Supports summary styles: `tldr`, `key_steps`, `contraindications`, `eligibility`
- Updated Streamlit **Summarize** page:
  - multi-doc selection
  - style selector
  - shows summary + citations table + expandable full snippets
  - stores `last_summary` + payload in session state
- Updated Streamlit **Evidence** page:
  - toggle between **Ask** vs **Summarize**
  - shows last payload + citations + snippets for the selected mode

### Day 5
- Added lightweight evaluation harness (`eval/`) to benchmark RAG quality + performance
- Metrics tracked:
  - latency (avg / p95 / max) for `/ask` and `/summarize`
  - citation coverage (fraction of responses returning citations)
  - grounding overlap (heuristic token overlap between output and retrieved snippets)
- Generates a timestamped JSON report in `eval/reports/`

### Day 6
- Dockerized the project:
  - `Dockerfile.api` (FastAPI + Uvicorn)
  - `Dockerfile.ui` (Streamlit UI)
  - `docker-compose.yml` to run **api + ui** together
  - `.dockerignore` to keep images small
- One-command local run: `docker compose up --build`

### Day 7
- Added CI with GitHub Actions:
  - `ruff check .`
  - `pytest` (tests folder scaffold)
- Cleaned up lint issues so `ruff` passes consistently

---

## Tech Stack
- **FastAPI** (API)
- **Streamlit** (UI)
- **Pydantic** (schemas)
- **pypdf** (PDF text extraction)
- **OpenAI** (chat + embeddings)
- **ChromaDB** (persistent vector store)
- **Docker / Docker Compose** (reproducible runtime)
- **GitHub Actions** (CI)

---

## Quickstart

### Option A: Run locally (venv)

#### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

#### 2) Install dependencies
```bash
pip install -U pip
pip install -e ".[dev]"
```

#### 3) Create `.env`
Create a `.env` file in project root:

```env
APP_NAME=guidelinecopilot-api
APP_VERSION=0.1.0
API_HOST=0.0.0.0
API_PORT=8000

MODEL_PROVIDER=openai
OPENAI_API_KEY=YOUR_KEY_HERE
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

DATA_DIR=data
MAX_UPLOAD_MB=30
```

#### 4) Run API
```bash
uvicorn apps.api.main:app --reload --port 8000
```

#### 5) Run Streamlit UI
```bash
streamlit run apps/ui/Home.py
```

Open:
- API health: http://localhost:8000/health
- API docs (Swagger): http://localhost:8000/docs
- UI: http://localhost:8501

---

### Option B: Run with Docker (recommended)

#### 1) Ensure `.env` exists
Same `.env` as above (Docker Compose loads it).

#### 2) Build + run
```bash
docker compose up --build
```

Open:
- UI: http://localhost:8501
- API: http://localhost:8000/docs

To stop:
```bash
docker compose down
```

---

## Demo Flow (2 minutes)
1) Open Streamlit UI → **Upload** → ingest a public guideline PDF
2) Go to **Ask** → select ingested `doc_id` → ask a question
3) Go to **Evidence** → view retrieved snippets + scores (traceability)
4) Go to **Summarize** → pick style → inspect citations/snippets
5) Evidence → toggle Ask/Summarize to audit the last run

---

## Evaluation (Day 5)

Run a small benchmark suite against the running API.

### Run
1) Start API (local or Docker):
```bash
uvicorn apps.api.main:app --reload --port 8000
```
(or `docker compose up --build`)

2) Run eval:
```bash
python -m eval.run_eval
```

Optional: point eval to a different base URL:
```bash
# Windows PowerShell example
$env:EVAL_BASE_URL="http://localhost:8000"
python -m eval.run_eval
```

### Output
- Prints summary metrics to console
- Saves a JSON report to `eval/reports/report_YYYYMMDD_HHMMSS.json`

### What it measures
- Latency: avg / p95 / max per endpoint
- Citation coverage: % responses that include citations
- Grounding overlap (heuristic): overlap between generated text and retrieved snippets

---

## API Overview

### `POST /ingest`
Uploads a guideline PDF, extracts text, chunks, embeds, and indexes into Chroma.

Returns:
- `doc_id`
- `pages`
- `chunks_indexed`
- `deduped` + `message` (if duplicate detected)

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
- `answer`
- `citations[]` with `doc_id`, `page`, `chunk_id`, `snippet`, `score`

### `POST /summarize`
Grounded summary over guideline chunks.

Example request:
```json
{
  "doc_ids": ["doc_1234abcd"],
  "style": "tldr"
}
```

Response includes:
- `summary`
- `citations[]`

---

## Notes / Current Limitations
- `/documents` registry is **in-memory** (resets on server restart). PDFs + Chroma persistence stay on disk.
- Only **public guideline PDFs** are supported (no patient data).
- Chunk quality depends on PDF text extraction quality (some PDFs contain headers/footers/license pages).

---

## Project Roadmap
- [x] Day 1: Ingest UI + doc registry + duplicate detection
- [x] Day 2: Text extraction + chunking + embeddings + retrieval + `/ask` with citations
- [x] Day 3: Ask UI polish + Evidence view (audit trail)
- [x] Day 4: Summarize endpoint + UI (grounded summaries + citations) + Evidence tab support
- [x] Day 5: Evaluation suite (latency + citation coverage + faithfulness checks)
- [x] Day 6: Dockerize (docker-compose: api + ui)
- [x] Day 7: CI (ruff + tests)
