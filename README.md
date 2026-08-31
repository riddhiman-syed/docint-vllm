# Document QA System

Self-hosted, agentic RAG pipeline for PDF question-answering — no paid LLM APIs anywhere in the stack. The LLM/VLM is served locally via [vLLM](https://github.com/vllm-project/vllm), retrieval runs against a self-hosted [Qdrant](https://qdrant.tech/) vector store, and open-source embeddings replace any external embedding API.

## Architecture

```
Ingestion:
  PDFs ──▶ parse text (PyMuPDF) ────────────▶ chunk (page-scoped) ──▶ embed (bge-base) ──▶ Qdrant
        └▶ extract embedded images ──▶ caption (VLM) ─┘

Query:
  Question ──▶ router ──▶ retrieve ──▶ grade ──▶ generate (vLLM, cited) ──▶ verify ──┬──▶ response
                  │                       │                                          │
                  ▼                       ▼ (nothing relevant, retries left)         ▼ (unsupported, retries left)
             direct answer          rewrite query ──▶ (loop back to retrieve)   regenerate ──▶ (loop)
                                       │ (retries exhausted)                      │ (retries exhausted)
                                       ▼                                          ▼
                                    fallback ("I don't know") ◀───────────────────┘

Visual Q&A (standalone): uploaded image + question ──▶ VLM ──▶ answer (no retrieval)
```

- **Serving**: vLLM, OpenAI-compatible endpoint. A single model (`Qwen2.5-VL`, AWQ-quantized) handles both text generation and vision — no separate VLM deployment needed.
- **Embeddings**: `sentence-transformers`, `BAAI/bge-base-en-v1.5`.
- **Vector store**: Qdrant, cosine similarity, `on_disk_payload` for citation metadata.
- **Orchestration**: LangGraph state machine with two independent retry budgets — retrieval (rewrite-and-retry when nothing relevant comes back) and generation (regenerate once if self-verification finds the answer unsupported by context). Falls back to an honest "I don't know" rather than hallucinating once retries are exhausted.
- **API**: FastAPI — `POST /ask`, `POST /ask-image`, `GET /health`.
- **Packaging**: native shell scripts bring up Qdrant, vLLM, and the API together (see [Bringing up the stack](#bringing-up-the-stack)).

## What's implemented

- [x] PDF parsing with per-page text extraction, page-accurate citations
- [x] Page-scoped chunking with stable chunk IDs (no cross-document text-collision bugs — see [Design notes](#design-notes))
- [x] Open-source embeddings + Qdrant indexing
- [x] Agentic retrieval graph: router, retrieve, grade, rewrite-retry, generate, self-verify, fallback
- [x] `POST /ask` — full RAG pipeline, returns answer + route taken + source citations
- [x] `GET /health` — independently reports Qdrant and vLLM reachability
- [x] Multimodal, both assignment options:
  - Image-aware ingestion: embedded PDF images are captioned by the VLM and indexed alongside text (`chunk_type: "image_caption"`), so chart/figure content is retrievable and cited like any other chunk
  - `POST /ask-image` — direct visual Q&A on an uploaded image, bypassing retrieval entirely
- [x] Native startup scripts (`scripts/*.sh`) bringing up the full stack together, health-checked end to end
- [ ] Part A throughput/latency benchmark — pending (see [Benchmark](#benchmark))

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in real values if defaults don't fit
```

You'll also need a local Qdrant install (see `QDRANT_BIN` in `scripts/start_qdrant.sh` if it's not at the default expected path).

## Bringing up the stack

```bash
chmod +x scripts/*.sh
./scripts/start_all.sh   # brings up Qdrant, vLLM, and the API in order, health-checking between each
./scripts/stop_all.sh    # clean shutdown
```

Individual services can also be started on their own via `scripts/start_qdrant.sh`, `scripts/start_vllm.sh`, and `scripts/start_api.sh`. All scripts detect already-running services (by health endpoint, not just PID tracking) and skip starting duplicates — safe to run against instances started manually or in a previous session. Logs land in `logs/{qdrant,vllm,api}.log`; PIDs are tracked in `.run/`.

Environment variables for tuning the vLLM launch (model, max sequence length, GPU memory utilization, etc.) are documented as overridable defaults directly in `scripts/start_vllm.sh`.

**Why scripts instead of Docker Compose**: the assignment allows either. Docker itself wasn't available in the development environment (no `dockerd`, and the container's isolation didn't support running a nested Docker daemon), so native scripts were used and verified end-to-end instead.

## Usage

**Ingest documents:**
```bash
python scripts/ingest.py --source pdf/ --recreate
```
Parses text and extracts+captions embedded images from every PDF under `--source`, embeds everything, and indexes into Qdrant. `--recreate` drops and rebuilds the collection; omit to add incrementally.

**Ask a question:**
```bash
curl -X POST http://localhost:8080/ask -H "Content-Type: application/json" -d '{"question": "What is PMSBY?"}'
```
Returns `{"answer": "...", "route": "retrieve", "sources": [{"source": "...", "page_number": N, "score": 0.59, "chunk_type": "text"}]}`.

**Ask about an uploaded image directly:**
```bash
curl -X POST http://localhost:8080/ask-image -F "question=What does this chart show?" -F "image=@/path/to/image.png"
```

**Check system health:**
```bash
curl http://localhost:8080/health
```

## Project structure

```
app/
  config.py                                       # env-based settings (pydantic-settings)
  main.py                                         # FastAPI app entrypoint
  ingestion/
    pdf_parser.py                                 # PDF -> per-page text
    image_extractor.py                            # PDF -> embedded raster images
    chunker.py                                    # page text -> overlapping chunks
    captioner.py                                  # images -> VLM-generated captions -> chunks
    embedder.py                                   # chunks/queries -> vectors (self-hosted)
    indexer.py                                    # vectors -> Qdrant
  llm/
    client.py                                     # vLLM OpenAI-compatible client wrapper (text + vision)
  rag/
    retriever.py                                  # Qdrant similarity search
    prompts.py                                    # centralized prompt templates
    graph.py                                      # LangGraph agentic workflow
  api/
    schemas.py                                    # request/response models
    routes.py                                     # /ask, /ask-image, /health
scripts/
  ingest.py                                       # CLI: parse -> chunk -> embed -> index (text + images)
  start_all.sh / stop_all.sh / start_*.sh         # native service orchestration
pdf/                                              # sample documents
```

## Design notes

- **Chunk identity**: each chunk gets a generated UUID rather than being keyed by its own text — an earlier version of this pipeline used text-as-key, which silently merged any two chunks with identical content (repeated boilerplate, headers) into one entry with a combined page list. Fixed by scoping every chunk to exactly the one page/image it came from.
- **Citations**: the generation prompt requires a `(source.pdf, p.N)` citation on every factual sentence, enforced with a concrete few-shot example in the prompt rather than an abstract instruction — the abstract version was unreliable at smaller model sizes during testing.
- **Self-verification**: the `verify` node re-checks the generated answer against the retrieved context for unsupported claims before returning it, independently of the citation-formatting fix above.
- **Image extraction scope**: only catches images embedded as raster files (photos, scanned charts saved as images) via PyMuPDF. Charts drawn with native PDF vector-graphics operators aren't captured — full-page rendering would be the extension needed for that.
- **Qdrant local install**: not vendored in this repo (`qdrant/` in `.gitignore`) — install separately.
- **PDF parsing library**: uses PyMuPDF rather than the more common `pypdf`, for extraction speed/quality and to enable future full-page rendering (`pypdf` can't do this). Worth noting: PyMuPDF is AGPL-3.0-licensed, versus `pypdf`'s permissive BSD license. AGPL's copyleft terms are triggered by network use, not just distribution — relevant here since this is a served API. A deliberate, known trade-off for this project; worth reconsidering if this codebase is ever repurposed beyond its original context.

## Benchmark

Not run — the GPU was occupied by an unrelated process for the remainder of development, and VRAM never became available again to run this. Once available, the intended methodology is:
```bash
vllm bench serve --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ --num-prompts <N> --request-rate <20-50 concurrent>
```
capturing tokens/sec, time-to-first-token, and P95 latency against the target 7B/AWQ config (not the smaller model used for interim development/debugging on constrained VRAM).

## Notes on the target hardware (RTX 5090 / Blackwell, sm_120)

This GPU generation needed a few non-obvious fixes during development, already baked into `scripts/start_vllm.sh`, documented here in case they're useful elsewhere:
- vLLM's default FlashInfer sampler and attention backend both fail on sm_120 with a misleading `"requires sm75"` error — actually a broken compatibility check in that package, not a real hardware limitation. Fix: `--attention-backend TRITON_ATTN` plus uninstalling `flashinfer-python` outright (env-var-only disabling proved unreliable across FlashInfer's different internal call sites).
- First boot after any change to model/quantization/shapes triggers a cold Triton kernel compile (no prebuilt cache exists yet for this architecture) — this can take 10-20+ minutes and produces no log output while it happens; a pegged CPU core with no crash is the normal signature of this, not a hang.
