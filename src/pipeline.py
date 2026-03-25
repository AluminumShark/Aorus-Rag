"""End-to-end RAG pipeline: query → retrieve → generate."""

from __future__ import annotations

from collections.abc import Iterator

from src.embedder import load_model, encode
from src.indexer import load_index, search
from src.chunker import load_chunks
from src.generator import load_llm, generate


def _detect_lang(text: str) -> str:
    """Detect language: any CJK character → Chinese, otherwise English."""
    return "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"


def load_pipeline() -> dict:
    """Load all components needed for the RAG pipeline."""
    return {
        "model": load_model(),
        "index": load_index(),
        "chunks": load_chunks(),
        "llm": load_llm(),
    }


def query(pipe: dict, user_query: str, top_k: int = 5) -> Iterator[str]:
    """Run the full RAG pipeline: embed → search → generate.

    Always retrieves top-k chunks and lets the LLM decide whether
    the context is relevant enough to answer.
    """
    query_vec = encode(pipe["model"], [user_query], is_query=True)

    scores, indices = search(pipe["index"], query_vec[0], top_k=top_k)

    lang = _detect_lang(user_query)

    context = "\n".join(pipe["chunks"][i]["text"] for i in indices)

    return generate(pipe["llm"], context, user_query, lang=lang)
