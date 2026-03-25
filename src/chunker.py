
"""Split parsed specs into chunks for embedding and retrieval."""

from __future__ import annotations

import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Bilingual aliases so Chinese queries match English category names in embedding space.
CATEGORY_ALIASES: dict[str, str] = {
    "OS": "作業系統",
    "CPU": "處理器",
    "Video Graphics": "顯示卡/顯卡/GPU",
    "Display": "螢幕/顯示器",
    "System Memory": "記憶體/RAM",
    "Storage": "儲存/硬碟/SSD",
    "Keyboard Type": "鍵盤",
    "I/O Port": "連接埠/接口",
    "Audio": "音效/喇叭",
    "Communications": "網路/WiFi/藍牙",
    "Webcam": "視訊鏡頭",
    "Security": "安全性",
    "Battery": "電池",
    "Adapter": "電源供應器/充電器",
    "Dimensions (W x D x H)": "尺寸/機身大小",
    "Weight": "重量",
    "Color": "顏色",
}


def _label(category: str) -> str:
    """Return 'Category (中文別名)' if alias exists, else just the category."""
    alias = CATEGORY_ALIASES.get(category)
    return f"{category} ({alias})" if alias else category


def _clean_value(value: str) -> str:
    """Replace newlines with ', ' for a cleaner single-line format."""
    return ", ".join(line.strip() for line in value.split("\n") if line.strip())


def create_chunks(all_specs: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Convert nested specs dict into a deduplicated list of text chunks.

    Shared specs (identical across all models) get one chunk without model prefix.
    Differing specs get one chunk per model with "[Model]" prefix.

    Example shared:  "CPU: Intel® Core™ Ultra 9 ..."
    Example differ:  "[AORUS MASTER 16 BZH] Video Graphics: RTX 5090 ..."
    """
    model_names = list(all_specs.keys())
    # Use the first model's categories as reference
    categories = list(all_specs[model_names[0]].keys())

    chunks: list[dict[str, str]] = []
    for category in categories:
        # Collect this category's value for each model
        values = {model: all_specs[model][category] for model in model_names}
        unique_values = set(values.values())

        if len(unique_values) == 1:
            # All models share the same value — one chunk with shared note
            clean = _clean_value(next(iter(unique_values)))
            models_str = "/".join(
                name.split()[-1] for name in model_names  # "BZH/BYH/BXH"
            )
            text = f"{_label(category)} (shared by {models_str}): {clean}"
            chunks.append({"category": category, "text": text})
        else:
            # Values differ — one chunk per model, with model prefix
            for model_name in model_names:
                clean = _clean_value(values[model_name])
                text = f"[{model_name}] {_label(category)}: {clean}"
                chunks.append({
                    "model": model_name,
                    "category": category,
                    "text": text,
                })
    return chunks

def save_chunks(chunks: list[dict[str, str]], filename: str = "chunks.json") -> Path:
    """Save chunks to data/processed/ as JSON."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / filename
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_chunks(filename: str = "chunks.json") -> list[dict[str, str]]:
    """Load previously saved chunks."""
    path = PROCESSED_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    from src.scraper import load_specs_json

    all_specs = load_specs_json()
    chunks = create_chunks(all_specs)
    save_chunks(chunks)
    print(f"Created {len(chunks)} chunks:")
    for chunk in chunks:
        preview = chunk["text"][:100]
        print(f"  {preview}...")
