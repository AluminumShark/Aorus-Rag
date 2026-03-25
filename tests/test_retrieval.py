"""Retrieval quality tests — uses real embedding model + FAISS.

These tests verify that semantic search retrieves the correct chunks
for various queries. They caught the top_k=3 GPU bug and prevent
regressions.

Marked @slow because they load the embedding model (~120MB, CPU).
Skip with: pytest -m "not slow"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunker import load_chunks
from src.embedder import load_model, encode
from src.indexer import build_index, search

CHUNKS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def retrieval_env() -> dict:
    """Load embedding model + chunks + build index (once per module)."""
    if not CHUNKS_PATH.exists():
        pytest.skip("data/processed/chunks.json not found (run build_index first)")

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    model = load_model()
    vectors = encode(model, texts)  # passage encoding (default)
    index = build_index(vectors)
    return {"model": model, "index": index, "chunks": chunks, "vectors": vectors}


def _search_chunks(env: dict, query: str, top_k: int = 5) -> list[dict]:
    """Helper: encode query, search, return matched chunks."""
    q_vec = encode(env["model"], [query], is_query=True)
    _, indices = search(env["index"], q_vec[0], top_k=top_k)
    return [env["chunks"][i] for i in indices]


def _gpu_chunk_models(results: list[dict]) -> set[str]:
    """Extract model names from Video Graphics chunks in results."""
    return {
        c["model"]
        for c in results
        if c.get("category") == "Video Graphics" and "model" in c
    }


class TestGPURetrieval:
    """The core test: GPU queries must retrieve all 3 model variants."""

    def test_gpu_query_english(self, retrieval_env: dict) -> None:
        results = _search_chunks(retrieval_env, "What GPU does it have?")
        models = _gpu_chunk_models(results)
        assert len(models) == 3, f"Expected 3 GPU models, got {models}"

    def test_gpu_query_chinese(self, retrieval_env: dict) -> None:
        """Chinese GPU query must retrieve all 3 GPU chunks.

        The e5-small model handles cross-lingual retrieval well,
        and chunk text includes bilingual aliases (顯示卡/顯卡/GPU).
        """
        results = _search_chunks(retrieval_env, "顯卡是什麼型號？")
        models = _gpu_chunk_models(results)
        assert len(models) == 3, f"Expected 3 GPU models, got {models}"

    def test_top_k_5_vs_3_gpu_coverage(self, retrieval_env: dict) -> None:
        """top_k=5 retrieves all 3 GPU chunks; top_k=3 should also get all 3."""
        results_5 = _search_chunks(retrieval_env, "GPU specs", top_k=5)
        results_3 = _search_chunks(retrieval_env, "GPU specs", top_k=3)

        models_5 = _gpu_chunk_models(results_5)
        models_3 = _gpu_chunk_models(results_3)

        assert len(models_5) == 3, f"top_k=5 should get all 3, got {models_5}"
        assert len(models_3) == 3, f"top_k=3 should get all 3, got {models_3}"


class TestSpecificRetrieval:
    def test_battery_query(self, retrieval_env: dict) -> None:
        results = _search_chunks(retrieval_env, "電池容量多大？")
        categories = {c.get("category") for c in results}
        assert "Battery" in categories

    def test_unrelated_query_low_score(self, retrieval_env: dict) -> None:
        """Completely unrelated query should get lower scores than relevant ones."""
        q_vec = encode(retrieval_env["model"], ["今天天氣如何？"], is_query=True)
        scores_unrelated, _ = search(retrieval_env["index"], q_vec[0], top_k=1)

        q_vec2 = encode(retrieval_env["model"], ["電池容量多大？"], is_query=True)
        scores_relevant, _ = search(retrieval_env["index"], q_vec2[0], top_k=1)

        assert scores_unrelated[0] < scores_relevant[0], (
            f"Unrelated ({scores_unrelated[0]:.4f}) should score lower "
            f"than relevant ({scores_relevant[0]:.4f})"
        )
