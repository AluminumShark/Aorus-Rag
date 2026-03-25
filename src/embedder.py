"""Encode texts into normalized vectors for semantic search."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"


def load_model() -> SentenceTransformer:
    """Load the multilingual sentence-transformer model."""
    return SentenceTransformer(MODEL_NAME)


def encode(model: SentenceTransformer, texts: list[str],
           is_query: bool = False) -> np.ndarray:
    """Encode texts into L2-normalized vectors.

    The e5 model family requires "query: " or "passage: " prefixes.
    Set is_query=True when encoding user queries.

    Returns shape (len(texts), embedding_dim) as float32.
    """
    prefix = "query: " if is_query else "passage: "
    prefixed = [prefix + t for t in texts]
    vectors = model.encode(prefixed, show_progress_bar=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / norms).astype(np.float32)

if __name__ == "__main__":
    from src.chunker import load_chunks

    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    model = load_model()
    vectors = encode(model, texts)

    print(f"Encoded {len(texts)} chunks → shape {vectors.shape}")
    print(f"Norm of first vector: {np.linalg.norm(vectors[0]):.4f}")  # should be 1.0
