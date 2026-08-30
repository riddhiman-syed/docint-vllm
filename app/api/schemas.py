from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")


class SourceCitation(BaseModel):
    source: str
    page_number: int
    score: float
    chunk_type: str = "text"
    chunk_type: str = "text"


class AskResponse(BaseModel):
    answer: str
    route: str
    sources: list[SourceCitation]


class ServiceStatus(BaseModel):
    status: str  # "ok" | "unreachable"
    detail: str | None = None


class AskImageResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str  # "ok" if all dependencies are healthy, "degraded" otherwise
    qdrant: ServiceStatus
    vllm: ServiceStatus
