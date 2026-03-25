"""One-shot script: scrape → chunk → embed → build FAISS index."""

from src.scraper import fetch_spec_page, parse_specs, save_raw_html, save_specs_json
from src.chunker import create_chunks, save_chunks
from src.embedder import load_model, encode
from src.indexer import build_index, save_index


def main() -> None:
    # 1. Scrape
    print("Fetching spec page...")
    html = fetch_spec_page()
    save_raw_html(html)
    specs = parse_specs(html)
    save_specs_json(specs)
    print(f" -> {len(specs)} models parsed")

    # 2. Chunk
    chunks = create_chunks(specs)
    save_chunks(chunks)
    print(f" -> {len(chunks)} chunks created")

    # 3. Embed
    print("Loading embedding model...")
    model = load_model()
    texts = [c["text"] for c in chunks]
    vectors = encode(model, texts)
    print(f" -> vectors shape: {vectors.shape}")

    # 4. Index
    index = build_index(vectors)
    path = save_index(index)
    print(f" -> index saved to {path}")

    print("Done!")


if __name__ == "__main__":
    main()
