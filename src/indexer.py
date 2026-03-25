"""FAISS index for vector similarity search."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from normalized vectors."""
    dim = vectors.shape[1]  # 384
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)  # ty: ignore[missing-argument]  # faiss SWIG stub mismatch
    return index


def search(
    index: faiss.IndexFlatIP, query_vec: np.ndarray, top_k: int = 3
) -> tuple[list[float], list[int]]:
    """Search the index and return (scores, indices) for top-k results."""
    scores, indices = index.search(query_vec.reshape(1, -1), top_k)  # ty: ignore[missing-argument]  # faiss SWIG stub mismatch
    return scores[0].tolist(), indices[0].tolist()


def save_index(index: faiss.IndexFlatIP, filename: str = "faiss.index") -> Path:
    """Save FAISS index to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_DIR / filename
    faiss.write_index(index, str(path))
    return path


def load_index(filename: str = "faiss.index") -> faiss.IndexFlatIP:
    """Load FAISS index from disk."""
    path = INDEX_DIR / filename
    return faiss.read_index(str(path))


if __name__ == "__main__":
    from src.chunker import load_chunks
    from src.embedder import load_model, encode

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    model = load_model()
    vectors = encode(model, texts)

    index = build_index(vectors)
    save_index(index)
    print(f"Built index with {index.ntotal} vectors, saved to {INDEX_DIR}")

    # Quick test: search with first chunk as query
    scores, indices = search(index, vectors[0])
    print(f"Test search — top match: chunk {indices[0]}, score {scores[0]:.4f}")
