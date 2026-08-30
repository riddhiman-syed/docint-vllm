import logging

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Document QA System",
    description="Self-hosted, agentic RAG API for PDF question-answering.",
    version="0.1.0",
)
app.include_router(router)
