# Agentic RAG — Architecture & Evaluation Notes

## What this is

`/ask/agentic` is a second retrieval path alongside `/ask`, built with LangGraph.
Instead of a single-shot hybrid search → generate, it wraps retrieval in a
self-correcting loop: an LLM grades whether the retrieved context is actually
sufficient before generation runs, and if not, rewrites the query and retries.

Both endpoints share the same `AskRequest`/`AskResponse` contract, the same
hybrid search + reranker (`core/retrieval/hybrid.py`), and the same generation
prompt (`ASK_SYSTEM`). The agentic layer is purely an orchestration change —
retrieval and generation logic are unmodified and untouched by this feature.

## Graph

```
START -> retrieve -> grade_documents --(yes)--> generate -> END
              ^               |
              |          (no, retries left)
              +---- rewrite_query
```

- **retrieve** — wraps `HybridRetriever.search()` (BM25 + vector + RRF + cross-encoder rerank), unchanged from `/ask`.
- **grade_documents** — LLM judge (`ChatOpenAI` + `.with_structured_output`) that scores retrieval "yes"/"no" on whether it's sufficient to answer the question. Strict by design — partial relevance scores "no".
- **rewrite_query** — reformulates the query from `original_question` (not the previous rewrite, to avoid compounding drift), only on the "no" path.
- **generate** — same `ASK_SYSTEM` prompt and context builder as `/ask`. Uses `original_question` so the answer addresses what the user actually asked, even after query rewrites.
- **decide_to_generate** — the conditional edge. Retry cap takes priority over the grade: once `MAX_RETRIES` is hit, it routes to `generate` regardless of grade, so the loop always terminates and falls back to `ASK_SYSTEM`'s existing "not covered" refusal rather than looping or erroring.

Code: `core/graph/`. Manual smoke test: `python -m scripts.run_agentic_rag "<question>"`.

## Test coverage

Unit tests cover the three places a bug would fail *silently* rather than
loudly — `decide_to_generate`'s retry-cap logic, `grade_documents`'s
yes/no classification, and `rewrite_query`'s use of `original_question`
(a regression here would quietly degrade answer quality without throwing
an error). `retrieve` and `generate` are intentionally left to integration-level
validation (RAGAS, below) rather than mocked unit tests, matching the existing
project convention of not unit-testing `answer_question()` either — generation
quality is validated by evaluation, not assertions on mocks.

| File | Covers |
|---|---|
| `tests/test_routing.py` | `decide_to_generate` — retry cap, grade routing |
| `tests/test_grade_documents.py` | `grade_documents` — empty-doc short circuit, yes/no scoring, prompt content |
| `tests/test_rewrite_query.py` | `rewrite_query` — rewrite content, retry increment, original-question anchoring |

## RAGAS evaluation

Same golden dataset (`eval/golden/who_hand_hygiene.json`, 20 questions, 5
categories), same four metrics, run via `eval/run_ragas_eval_agentic.py`
against `/ask/agentic`. Deliberately does **not** write to
`eval/baseline/ragas_baseline.json` — that file is read by CI's
`ragas-threshold-gate` and represents the `/ask` pipeline; overwriting it
would silently change what CI is gating on.

### Results (2026-07-28)

Compared against the actual current `/ask` baseline (post-reranker, `eval/baseline/ragas_baseline.json`) — **not** the pre-reranker baseline from an earlier stage of this project, which this doc mistakenly used in an earlier draft.

| Metric | Baseline `/ask` | Agentic (`MAX_RETRIES=2`) | Agentic (`MAX_RETRIES=1`, shipped) |
|---|---|---|---|
| faithfulness | 0.726 | 0.822 | 0.805 |
| answer_relevancy | 0.730 | 0.780 | 0.781 |
| context_precision | 0.634 | 0.659 | 0.669 |
| context_recall | 0.792 | 0.792 | 0.842 |
| **Passing (of 4)** | 4/4 (already passing pre-agentic) | 4/4 | 4/4 |

The baseline `/ask` pipeline was already passing all four thresholds after the
Day 13 reranker work — agentic retrieval doesn't flip any metric from fail to
pass. What it does is improve all four by roughly 5–11% on top of an
already-strong baseline: faithfulness +0.079, answer_relevancy +0.051,
context_precision +0.035, context_recall +0.050 (retries=1 config).
`MAX_RETRIES=1` matched or beat `MAX_RETRIES=2` on 3 of 4 metrics — the
quality gain from self-correction saturates after one rewrite; a second
retry attempt did not earn its additional cost. See ablation below.

**n=20 caveat:** the golden dataset is small. Treat these as directionally
strong evidence, not tight-confidence-interval results.

**Unanswerable questions remain a structural RAGAS limitation, not a
retrieval problem** (documented in `eval/RAGAS_SETUP.md`): correct refusal
behavior scores 0.00 on `answer_relevancy`/`context_precision` regardless
of retry count, because RAGAS has no way to reward "correctly declined to
answer." The agentic loop cannot fix this — it's a framework ceiling, not
a bug in this feature.

## Latency / retry-count ablation

Compared via `python -m scripts.compare_endpoints`, which calls both
endpoints for every golden-dataset question and reads each response's
server-reported `meta.latency_ms` (and `meta.retries` for the agentic side).

### Overall

| | Baseline `/ask` | Agentic, `MAX_RETRIES=2` | Agentic, `MAX_RETRIES=1` |
|---|---|---|---|
| Avg latency | 4,353 ms | 6,682 ms | 6,575 ms |
| Overhead vs baseline | — | +53.5% | +61.9%* |
| Total retries fired (of 20 questions) | — | 10 | 6 |

*The retries=1 overhead % looks worse only because baseline `/ask` ran
marginally faster on that particular rerun — normal API latency variance,
not a real effect of the retry cap.

### Where the cost concentrates — `unanswerable` questions

| | `MAX_RETRIES=2` | `MAX_RETRIES=1` |
|---|---|---|
| Avg latency | 11,042 ms | 7,938 ms (**-28%**) |
| Retries fired | 8 (all 4 questions hit the cap) | 4 (all 4 questions hit the cap) |
| context_recall | 0.625 | 0.750 |

Every `unanswerable` question exhausts its retry budget — there's
genuinely no relevant content for the grader to find, so it correctly
says "no" every time until the cap forces a fallback to `generate()`'s
existing refusal. This is the category paying the largest latency tax,
and it's exactly where a second retry adds cost without any measurable
RAGAS benefit (see the framework limitation above — refusals score 0.00
either way).

**Also worth noting:** even questions with *zero* retries pay a fixed
latency cost (e.g. `contraindication`: 0 retries, still +43% vs baseline)
— one `grade_documents` LLM call happens before every generation,
retry or not. The agentic layer is not free even in the best case.

### Decision: `MAX_RETRIES=1`

Shipped default, based on the ablation above: retries=1 captured
essentially all the measured quality gain over `/ask` while cutting
worst-case (`unanswerable`) latency by ~28% relative to retries=2. The
second retry attempt was not earning its cost on this corpus.

## Known limitations / next steps

- **doc_ids / mode not wired into the graph** — `/ask/agentic` always
  does a corpus-wide search (`doc_id=None`). Fine for the current
  single-document demo; would need threading through `GraphState` if a
  second document is added.
- **No query decomposition** — multi-part questions (the `synthesis`
  category) still show the weakest `context_recall` (0.75) of any
  category. A single query rewrite reformulates but doesn't split a
  genuinely multi-part question; true decomposition (parallel sub-query
  retrieval + synthesis) would be a separate feature targeting this
  specific gap.
- **Retriever rebuilt per request** — `retrieve()` mirrors `answer_question()`'s
  existing pattern of constructing `ChromaVectorStore`/`BM25Store`/`HybridRetriever`
  fresh on every call. Not introduced by this feature, but worth
  revisiting for both endpoints if latency becomes a priority.