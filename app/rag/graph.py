"""
Agentic RAG workflow, built as a LangGraph state machine:

    route ─┬─▶ direct_answer ──────────────────────────────▶ END
           └─▶ retrieve ──▶ grade ─┬─▶ generate ──▶ verify ─┬─▶ END
                             ▲     └─▶ rewrite ──▶ (loop)    └─▶ generate (retry)
                             └── (no relevant docs, retries left) ──┘
           (retries exhausted at either point) ──▶ fallback ──▶ END

Two independent retry budgets: retrieval (rewrite-and-retry when nothing
relevant comes back) and generation (retry once if self-verification
finds the answer isn't supported by the context).
"""
import json
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.llm.client import get_llm_client
from app.rag.prompts import (
    FALLBACK_ANSWER,
    GENERATE_SYSTEM,
    GENERATE_USER_TEMPLATE,
    GRADE_SYSTEM,
    GRADE_USER_TEMPLATE,
    REWRITE_SYSTEM,
    ROUTE_SYSTEM,
    VERIFY_SYSTEM,
    VERIFY_USER_TEMPLATE,
    format_context,
)
from app.rag.retriever import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

MAX_RETRIEVE_RETRIES = 2
MAX_GENERATE_RETRIES = 1
TOP_K = 5


class GraphState(TypedDict):
    original_question: str
    question: str
    route: str
    documents: list[RetrievedChunk]
    generation: str
    retrieve_retries: int
    generate_retries: int


def _parse_json_bool(raw: str, key: str, default: bool = False) -> bool:
    try:
        return bool(json.loads(raw).get(key, default))
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse JSON from LLM (%r), defaulting %s=%s", raw[:200], key, default)
        return default


# --- Nodes ---

def route_question(state: GraphState) -> dict:
    llm = get_llm_client()
    raw = llm.chat_json([
        {"role": "system", "content": ROUTE_SYSTEM},
        {"role": "user", "content": state["original_question"]},
    ])
    try:
        route = json.loads(raw).get("route", "retrieve")
    except json.JSONDecodeError:
        route = "retrieve"
    return {"route": route if route in ("retrieve", "direct") else "retrieve"}


def direct_answer(state: GraphState) -> dict:
    llm = get_llm_client()
    text = llm.chat([{"role": "user", "content": state["original_question"]}])
    return {"generation": text}


def retrieve_node(state: GraphState) -> dict:
    docs = retrieve(state["question"], top_k=TOP_K)
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    llm = get_llm_client()
    relevant = []
    for doc in state["documents"]:
        raw = llm.chat_json([
            {"role": "system", "content": GRADE_SYSTEM},
            {"role": "user", "content": GRADE_USER_TEMPLATE.format(question=state["question"], passage=doc.text)},
        ])
        if _parse_json_bool(raw, "relevant"):
            relevant.append(doc)
    logger.info("Graded %d/%d retrieved chunks as relevant", len(relevant), len(state["documents"]))
    return {"documents": relevant}


def rewrite_query(state: GraphState) -> dict:
    llm = get_llm_client()
    rewritten = llm.chat([
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": state["question"]},
    ])
    return {"question": rewritten.strip(), "retrieve_retries": state["retrieve_retries"] + 1}


def generate(state: GraphState) -> dict:
    llm = get_llm_client()
    context = format_context(state["documents"])
    text = llm.chat([
        {"role": "system", "content": GENERATE_SYSTEM},
        {"role": "user", "content": GENERATE_USER_TEMPLATE.format(context=context, question=state["original_question"])},
    ])
    return {"generation": text, "generate_retries": state["generate_retries"] + 1}


def verify(state: GraphState) -> dict:
    # No-op pass-through node; the actual decision happens in route_after_verify.
    # Kept as a distinct node so it shows up as its own step in the graph/trace.
    return {}


def fallback(state: GraphState) -> dict:
    return {"generation": FALLBACK_ANSWER}


# --- Conditional edges ---

def route_after_route_question(state: GraphState) -> str:
    return "direct_answer" if state["route"] == "direct" else "retrieve"


def route_after_grade(state: GraphState) -> str:
    if state["documents"]:
        return "generate"
    if state["retrieve_retries"] < MAX_RETRIEVE_RETRIES:
        return "rewrite"
    return "fallback"


def route_after_verify(state: GraphState) -> str:
    llm = get_llm_client()
    context = format_context(state["documents"])
    raw = llm.chat_json([
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": VERIFY_USER_TEMPLATE.format(context=context, answer=state["generation"])},
    ])
    supported = _parse_json_bool(raw, "supported", default=True)  # fail open: don't loop forever on a flaky grader
    if supported:
        return "end"
    if state["generate_retries"] < MAX_GENERATE_RETRIES:
        return "generate"
    return "fallback"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route_question", route_question)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_documents)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("fallback", fallback)

    graph.set_entry_point("route_question")
    graph.add_conditional_edges("route_question", route_after_route_question, {"direct_answer": "direct_answer", "retrieve": "retrieve"})
    graph.add_edge("direct_answer", END)
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges("verify", route_after_verify, {"end": END, "generate": "generate", "fallback": "fallback"})
    graph.add_edge("fallback", END)

    return graph.compile()


def answer_question(question: str) -> dict:
    """Entry point: run the full graph for a question, return the final state."""
    app = build_graph()
    initial_state: GraphState = {
        "original_question": question,
        "question": question,
        "route": "",
        "documents": [],
        "generation": "",
        "retrieve_retries": 0,
        "generate_retries": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state
