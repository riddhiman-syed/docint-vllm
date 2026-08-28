# Document QA System

Self-hosted, agentic RAG pipeline for PDF question-answering — no paid LLM APIs. The LLM/VLM is served locally via [vLLM](https://github.com/vllm-project/vllm), retrieval runs against a self-hosted [Qdrant](https://qdrant.tech/) vector store, and open-source embeddings replace any external embedding API.

> **Status: work in progress.** This README tracks what's built so far. Sections for the agentic workflow, API, Docker packaging, and benchmarks will be filled in as those pieces land.

## Architecture (planned)

```
PDFs ──▶ parse (pypdf) ──▶ chunk (page-scoped) ──▶ embed (bge-base) ──▶ Qdrant
                                                                          │
User question ──▶ router ──▶ retrieve ──▶ grade ──▶ answer (vLLM) ──▶ verify ──▶ response
                                 │                                        │
                                 └──────────── rewrite & retry ◀──────────┘
```

- **Serving**: vLLM, OpenAI-compatible endpoint, single self-hosted model for both text and image (VLM) generation
- **Embeddings**: `sentence-transformers`, `BAAI/bge-base-en-v1.5` by default
- **Vector store**: Qdrant
- **Orchestration**: LangGraph agentic workflow (router → retrieve+grade → answer+verify → fallback) — *not yet implemented*
- **API**: FastAPI `/ask` + `/health` — *not yet implemented*
- **Packaging**: Docker Compose (vLLM + Qdrant + API) — *not yet implemented*

## What's implemented

- [x] PDF parsing with per-page text extraction (`app/ingestion/pdf_parser.py`)
- [x] Page-scoped chunking with stable chunk IDs (`app/ingestion/chunker.py`)
- [x] Open-source embedding generation (`app/ingestion/embedder.py`)
- [x] Qdrant indexing with citation metadata (`app/ingestion/indexer.py`)
- [x] CLI ingestion entrypoint (`scripts/ingest.py`)
- [ ] Agentic retrieval/answer graph
- [ ] FastAPI serving layer
- [ ] Multimodal (image) ingestion and Q&A
- [ ] Docker Compose packaging
- [ ] Throughput/latency benchmark

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in real values
```

You'll also need a running Qdrant instance:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

And a running vLLM server (see `.env.example` for the expected `VLLM_BASE_URL`/`VLLM_MODEL_NAME`).

## Usage

Ingest a directory of PDFs into Qdrant:

```bash
python scripts/ingest.py --source pdf/ --recreate
```

- `--source`: directory to scan for PDFs (recursive)
- `--recreate`: drop and rebuild the Qdrant collection from scratch; omit to add to an existing collection

## Project structure

```
app/
  config.py             # env-based settings (pydantic-settings)
  ingestion/
    pdf_parser.py        # PDF → per-page text
    chunker.py            # page text → overlapping chunks
    embedder.py            # chunks/queries → vectors (self-hosted)
    indexer.py              # vectors → Qdrant
  rag/                       # agentic retrieval workflow (WIP)
  llm/                       # vLLM client wrapper (WIP)
  api/                        # FastAPI routes (WIP)
scripts/
  ingest.py                    # CLI: parse → chunk → embed → index
pdf/                            # sample documents for testing
```

## Notes

- Chunks are scoped to a single source page each, with a generated UUID as the identifier — no chunk is ever merged with another based on matching text content.
- Embedding queries and documents are encoded differently: queries get the `bge` instruction prefix, indexed documents don't, per the model's training setup.
