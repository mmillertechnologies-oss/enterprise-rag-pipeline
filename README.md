# Enterprise RAG Pipeline

Production-grade retrieval-augmented generation system for enterprise document Q&A. Built for real workloads — async ingestion, chunking strategies tuned for accuracy, hybrid retrieval, and a REST API ready to integrate into existing systems.

## Architecture

```
Documents (txt, md, py, json, csv)
        │
        ▼
┌──────────────────┐
│  Document Loader │  sha256 doc IDs, encoding-safe, skips empty files
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  Recursive Chunker   │  512-token chunks, 64-token overlap, natural boundaries
└────────┬─────────────┘
         │
         ▼
┌─────────────────────────┐
│  OpenAI Embedder        │  text-embedding-3-small, batch 100, retry on rate limit
└────────┬────────────────┘
         │
         ▼
┌──────────────────┐
│  ChromaDB Store  │  cosine similarity, persistent, incremental upsert
└────────┬─────────┘
         │
    query │
         ▼
┌──────────────────────────┐
│  VectorRetriever         │  top-k with score threshold filtering
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  RAGGenerator (GPT-4)    │  structured prompt, source citations, temp=0.1
└────────┬─────────────────┘
         │
         ▼
    FastAPI REST API  →  /query  /ingest  /status  /health
```

## Demo

**Ingesting a document directory (247 files, ~18MB):**

```
$ python scripts/ingest.py --dir ./docs/

INFO Indexed knowledge-base/onboarding.md — 14 chunks
INFO Indexed knowledge-base/infrastructure-runbook.md — 31 chunks
INFO Indexed knowledge-base/incident-response.md — 22 chunks
INFO Indexed knowledge-base/api-reference.md — 47 chunks
INFO Indexed knowledge-base/architecture-overview.md — 28 chunks
...
INFO Done. 1,842 total chunks indexed across 247 documents.
INFO Index size: 1842 chunks
```

**Querying — natural language over 1,800+ chunks in ~180ms:**

```
$ curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the escalation process for a Sev1 storage outage?", "k": 5}' \
    | python -m json.tool
```

```json
{
  "answer": "For a Sev1 storage outage, the escalation process is as follows:\n\n1. **Immediate (0-5 min):** On-call engineer acknowledges the PagerDuty alert and opens a war room bridge. Notify the Storage Operations lead via the #sev1-incidents Slack channel.\n2. **Triage (5-15 min):** Run `storage-tiering.ps1 -DryRun` to assess array health. Check Azure Monitor for IOPS and latency anomalies across all five arrays.\n3. **Escalate (15 min if unresolved):** Page the senior storage engineer and open a Microsoft support ticket at severity A. Document all actions in the incident log.\n4. **Executive notification (30 min if unresolved):** Engineering manager notifies VP of Infrastructure. Update the status page.\n\nSource: `incident-response.md` — Section 4.2, Sev1 Runbook",
  "sources": [
    "docs/knowledge-base/incident-response.md",
    "docs/knowledge-base/infrastructure-runbook.md"
  ],
  "chunks_retrieved": 5,
  "tokens_used": 934
}
```

**Index status:**

```
$ curl -s http://localhost:8000/status | python -m json.tool

{
  "status": "ok",
  "documents_indexed": 1842,
  "model": "gpt-4-turbo-preview",
  "embedding_model": "text-embedding-3-small"
}
```

---

## Quick Start

```bash
git clone https://github.com/mmillertechnologies-oss/enterprise-rag-pipeline.git
cd enterprise-rag-pipeline

python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Start the API
uvicorn src.api.routes:app --reload

# Or with Docker
docker-compose up
```

## Ingest Documents

```bash
# Ingest a directory
python scripts/ingest.py --dir ./your-docs/

# Ingest a single file
python scripts/ingest.py --file ./report.md
```

## API Usage

**Query your documents:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the backup procedures?", "k": 5}'
```

**Response:**
```json
{
  "answer": "Based on the runbook (dr-procedures.md): backup validation runs nightly at 02:00...",
  "sources": ["./docs/dr-procedures.md"],
  "chunks_retrieved": 5,
  "tokens_used": 847
}
```

**Ingest via API:**
```bash
curl -X POST "http://localhost:8000/ingest/directory?path=./docs"
```

**Status:**
```bash
curl http://localhost:8000/status
# {"status":"ok","documents_indexed":1842,"model":"gpt-4-turbo-preview",...}
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4-turbo-preview` | Generation model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RETRIEVAL_K` | `5` | Chunks to retrieve per query |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.3` | Min cosine similarity score |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Vector store path |

## Design Decisions

**Chunking strategy:** Recursive splitting on paragraph → sentence → word boundaries keeps semantically coherent chunks. Mid-sentence splits significantly hurt retrieval quality.

**Score threshold:** Results below 0.3 cosine similarity are filtered rather than returned — a "not found" response is more useful than a hallucinated one.

**Incremental indexing:** ChromaDB upsert on doc_id + chunk_index means re-ingesting updated files doesn't duplicate chunks.

**Retry on rate limit:** Embedder retries with exponential backoff (5s, 10s, 20s) — essential for large batch ingestion jobs.

## Project Structure

```
src/
├── config.py              — Pydantic settings, env-driven
├── ingestion/
│   ├── loader.py          — File loading with sha256 IDs
│   └── chunker.py         — Recursive text splitter
├── embeddings/
│   └── embedder.py        — OpenAI batch embedder with retry
├── retrieval/
│   └── retriever.py       — ChromaDB index + query
├── generation/
│   └── generator.py       — GPT-4 with source citation prompt
└── api/
    ├── routes.py          — FastAPI routes
    └── schemas.py         — Pydantic request/response models
scripts/
└── ingest.py              — CLI ingest tool
```

## Requirements

- Python 3.11+
- OpenAI API key
- Docker (optional)
