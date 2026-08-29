"""
Prompt templates for each node in the agentic graph. Centralized here
so wording/format changes don't require touching graph logic.
"""

ROUTE_SYSTEM = """You decide whether a user's question needs document retrieval to answer, \
or can be handled directly.

Respond with JSON only: {"route": "retrieve"} or {"route": "direct"}.

Use "direct" only for greetings, chit-chat, or questions that are clearly unrelated to any \
document knowledge base (e.g. "hello", "what can you do?"). Use "retrieve" for anything that \
could plausibly be answered from documents, including if you're unsure."""

GRADE_SYSTEM = """You grade whether a retrieved passage is relevant to a user's question.

Respond with JSON only: {"relevant": true} or {"relevant": false}.

Grade "true" if the passage contains any information that would help answer the question, \
even partially. Grade "false" only if the passage is clearly unrelated."""

GRADE_USER_TEMPLATE = """Question: {question}

Passage:
{passage}"""

REWRITE_SYSTEM = """You rewrite a user's question to improve retrieval from a vector search \
index, when the original question failed to retrieve relevant documents.

Respond with the rewritten question only, no explanation, no quotes."""

GENERATE_SYSTEM = """You answer questions using only the provided context passages. \
Cite the source and page number for each claim, like: (source.pdf, p.3).

If the context doesn't contain enough information to answer, say so plainly instead of \
guessing or using outside knowledge."""

GENERATE_USER_TEMPLATE = """Context:
{context}

Question: {question}"""

VERIFY_SYSTEM = """You check whether an answer is fully supported by the given context, with \
no unsupported claims (no hallucination) and no missing citations for factual statements.

Respond with JSON only: {"supported": true} or {"supported": false}."""

VERIFY_USER_TEMPLATE = """Context:
{context}

Answer to check:
{answer}"""

FALLBACK_ANSWER = (
    "I couldn't find reliable information in the available documents to answer that question. "
    "Could you rephrase it, or ask about something covered in the ingested documents?"
)


def format_context(chunks) -> str:
    """chunks: list of RetrievedChunk"""
    return "\n\n".join(f"[{c.source}, p.{c.page_number}]\n{c.text}" for c in chunks)
