@"
# RAGAS Setup Notes

## Version
ragas==0.2.15 is pinned in pyproject.toml.

## Required manual patch
After installing, ragas/llms/base.py imports ChatVertexAI and VertexAI from
langchain_community which is not installed. Patch it out:

``''bash
# Remove VertexAI imports (not needed — we use OpenAI only)
pip show ragas | Select-String Location
``''

Then open .venv/Lib/site-packages/ragas/llms/base.py and remove these lines:
  from langchain_community.chat_models.vertexai import ChatVertexAI
  from langchain_community.llms import VertexAI
  ChatVertexAI,
  VertexAI,

## Running the eval
``''bash
# Load API key (PowerShell)
`$env:OPENAI_API_KEY = (Get-Content .env | Select-String 'OPENAI_API_KEY' | ForEach-Object { `$_ -replace 'OPENAI_API_KEY=', '' })

# Run
python -m eval.run_ragas_eval
``''

## Baseline scores (first run, 2026-05-31)
| Metric             | Score | Threshold | Status |
|--------------------|-------|-----------|--------|
| faithfulness       | 0.33  | 0.80      | FAIL   |
| answer_relevancy   | 0.68  | 0.70      | FAIL   |
| context_precision  | 0.40  | 0.60      | FAIL   |
| context_recall     | 0.57  | 0.55      | PASS   |
"@ | Set-Content eval\RAGAS_SETUP.md