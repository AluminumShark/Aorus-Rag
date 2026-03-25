"""Unit tests for src.indexer — uses fake vectors, no embedding model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.indexer import build_index, search, save_index, load_index


def _random_normalized_vectors(n: int, dim: int = 384) -> np.ndarray:
    """Generate n random L2-normalized vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


class TestBuildIndex:
    def test_index_size(self) -> None:
        vecs = _random_normalized_vectors(10)
        index = build_index(vecs)
        assert index.ntotal == 10

    def test_dimension(self) -> None:
        vecs = _random_normalized_vectors(5, dim=128)
        index = build_index(vecs)
        assert index.d == 128


class TestSearch:
    def test_self_search_score_near_one(self) -> None:
        vecs = _random_normalized_vectors(10)
        index = build_index(vecs)
        scores, indices = search(index, vecs[0], top_k=1)
        assert indices[0] == 0
        assert scores[0] == pytest.approx(1.0, abs=1e-4)

    def test_top_k_count(self) -> None:
        vecs = _random_normalized_vectors(20)
        index = build_index(vecs)
        scores, indices = search(index, vecs[0], top_k=5)
        assert len(scores) == 5
        assert len(indices) == 5

    def test_scores_descending(self) -> None:
        vecs = _random_normalized_vectors(20)
        index = build_index(vecs)
        scores, _ = search(index, vecs[0], top_k=5)
        assert scores == sorted(scores, reverse=True)


class TestSearchEdgeCases:
    def test_top_k_exceeds_index_size(self) -> None:
        """top_k larger than index size should not crash."""
        vecs = _random_normalized_vectors(5)
        index = build_index(vecs)
        scores, indices = search(index, vecs[0], top_k=10)
        assert len(scores) == 10
        assert len(indices) == 10

    def test_single_vector_index(self) -> None:
        """Index with just 1 vector should work."""
        vecs = _random_normalized_vectors(1)
        index = build_index(vecs)
        scores, indices = search(index, vecs[0], top_k=1)
        assert indices[0] == 0
        assert scores[0] == pytest.approx(1.0, abs=1e-4)


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path: Path) -> None:
        vecs = _random_normalized_vectors(10)
        index = build_index(vecs)
        save_index(index, filename="test.index")

        loaded = load_index(filename="test.index")
        assert loaded.ntotal == 10

        # Same search results after reload
        s1, i1 = search(index, vecs[0], top_k=3)
        s2, i2 = search(loaded, vecs[0], top_k=3)
        assert i1 == i2
