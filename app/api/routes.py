import logging

import httpx
from fastapi import APIRouter, HTTPException

from app.api.schemas import AskRequest, AskResponse, HealthResponse, ServiceStatus, SourceCitation
from app.config import get_settings
from app.ingestion.indexer import get_qdrant_client
from app.rag.graph import answer_question

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        state = answer_question(request.question)
    except Exception:
        logger.exception("Failed to answer question: %r", request.question)
        raise HTTPException(status_code=502, detail="Failed to reach the LLM or vector store — check /health")

    sources = [
        SourceCitation(source=d.source, page_number=d.page_number, score=d.score)
        for d in state["documents"]
    ]
    return AskResponse(answer=state["generation"], route=state["route"], sources=sources)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()

    qdrant_status = ServiceStatus(status="ok")
    try:
        get_qdrant_client().get_collections()
    except Exception as e:
        qdrant_status = ServiceStatus(status="unreachable", detail=str(e))

    vllm_status = ServiceStatus(status="ok")
    try:
        resp = httpx.get(f"{settings.VLLM_BASE_URL}/models", timeout=3.0)
        resp.raise_for_status()
    except Exception as e:
        vllm_status = ServiceStatus(status="unreachable", detail=str(e))

    overall = "ok" if qdrant_status.status == "ok" and vllm_status.status == "ok" else "degraded"
    return HealthResponse(status=overall, qdrant=qdrant_status, vllm=vllm_status)
