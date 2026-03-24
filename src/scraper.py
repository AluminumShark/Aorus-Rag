"""Scrape and parse GIGABYTE AORUS MASTER 16 AM6H product spec page."""

from __future__ import annotations

import json
from pathlib import Path
from bs4 import BeautifulSoup


SPEC_URL = "https://www.gigabyte.com/Laptop/AORUS-MASTER-16-AM6H/sp"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_spec_page(url: str = SPEC_URL) -> str:
    """Fetch product spec page HTML using Playwright (headless=False to bypass WAF)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        html = page.content()
        browser.close()
    return html


MODEL_NAMES = [
    "AORUS MASTER 16 BZH",
    "AORUS MASTER 16 BYH",
    "AORUS MASTER 16 BXH",
]


def parse_specs(html: str) -> dict[str, dict[str, str]]:
    """Parse spec key-value pairs for ALL model variants from the GIGABYTE page.

    Uses the desktop multiple-spec-content-wrapper which contains:
      - spec-column: category names (multiple-title divs)
      - content-column: 3 swiper-slides, one per model variant

    Returns a nested dict keyed by model name:
        {
            "AORUS MASTER 16 BZH": {"OS": "...", "CPU": "...", ...},
            "AORUS MASTER 16 BYH": {"OS": "...", "CPU": "...", ...},
            "AORUS MASTER 16 BXH": {"OS": "...", "CPU": "...", ...},
        }
    """
    soup = BeautifulSoup(html, "html.parser")

    wrapper = soup.find("div", class_="multiple-spec-content-wrapper")
    if not wrapper:
        raise ValueError("Could not find multiple-spec-content-wrapper. Page structure may have changed.")

    # Extract category names from spec-column
    spec_col = wrapper.find("div", class_="spec-column")
    categories = [
        t.get_text(strip=True)
        for t in spec_col.find_all("div", class_="multiple-title")
    ]

    # Extract values for each model variant from swiper slides
    content_col = wrapper.find("div", class_="content-column")
    slides = content_col.find_all("div", class_="swiper-slide")

    all_specs: dict[str, dict[str, str]] = {}
    for i, slide in enumerate(slides):
        model_name = MODEL_NAMES[i] if i < len(MODEL_NAMES) else f"Variant {i}"
        items = slide.find_all("div", class_="spec-item-list")
        specs: dict[str, str] = {}
        for j, item in enumerate(items):
            key = categories[j] if j < len(categories) else f"Unknown_{j}"
            value = item.get_text(separator="\n", strip=True)
            specs[key] = value
        all_specs[model_name] = specs

    if not all_specs:
        raise ValueError("Parsed 0 model variants. Check HTML structure.")

    return all_specs


def save_raw_html(html: str, filename: str = "spec_page.html") -> Path:
    """Save raw HTML to data/raw/ for reproducibility."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    path.write_text(html, encoding="utf-8")
    return path


def save_specs_json(specs: dict[str, dict[str, str]], filename: str = "specs.json") -> Path:
    """Save parsed specs dict to data/raw/ as JSON."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_specs_json(filename: str = "specs.json") -> dict[str, dict[str, str]]:
    """Load previously saved specs JSON."""
    path = RAW_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    # Quick test: fetch, parse, and save
    print(f"Fetching {SPEC_URL} ...")
    html = fetch_spec_page()
    save_raw_html(html)
    print("Saved raw HTML to data/raw/spec_page.html")

    all_specs = parse_specs(html)
    save_specs_json(all_specs)
    for model_name, specs in all_specs.items():
        print(f"\n{model_name} ({len(specs)} specs):")
        for key, value in specs.items():
            preview = value[:80].replace("\n", " ")
            print(f"  {key}: {preview}")
