"""Unit tests for src.scraper.parse_specs — pure HTML parsing, no network."""

from __future__ import annotations

import pytest

from src.scraper import parse_specs


def _build_html(categories: list[str], slides: list[list[str]]) -> str:
    """Build minimal HTML that mimics the GIGABYTE spec page structure."""
    titles = "\n".join(f'<div class="multiple-title">{cat}</div>' for cat in categories)
    slide_divs = []
    for values in slides:
        items = "\n".join(f'<div class="spec-item-list">{v}</div>' for v in values)
        slide_divs.append(f'<div class="swiper-slide">{items}</div>')

    return f"""
    <html><body>
    <div class="multiple-spec-content-wrapper">
        <div class="spec-column">{titles}</div>
        <div class="content-column">{"".join(slide_divs)}</div>
    </div>
    </body></html>
    """


class TestParseSpecs:
    def test_parses_three_models(self) -> None:
        html = _build_html(
            ["OS", "CPU"],
            [
                ["Windows 11", "Intel i9"],
                ["Windows 11", "Intel i9"],
                ["Windows 11", "Intel i9"],
            ],
        )
        specs = parse_specs(html)
        assert len(specs) == 3
        assert "AORUS MASTER 16 BZH" in specs
        assert specs["AORUS MASTER 16 BZH"]["OS"] == "Windows 11"
        assert specs["AORUS MASTER 16 BZH"]["CPU"] == "Intel i9"

    def test_missing_wrapper_raises(self) -> None:
        html = "<html><body><p>No spec table here</p></body></html>"
        with pytest.raises(ValueError, match="multiple-spec-content-wrapper"):
            parse_specs(html)

    def test_no_slides_raises(self) -> None:
        html = """
        <html><body>
        <div class="multiple-spec-content-wrapper">
            <div class="spec-column">
                <div class="multiple-title">OS</div>
            </div>
            <div class="content-column"></div>
        </div>
        </body></html>
        """
        with pytest.raises(ValueError, match="0 model variants"):
            parse_specs(html)

    def test_multiline_values_preserved(self) -> None:
        html = _build_html(
            ["Display"],
            [
                ["16 inch\nOLED\n240Hz"],
                ["16 inch\nOLED\n240Hz"],
                ["16 inch\nOLED\n240Hz"],
            ],
        )
        specs = parse_specs(html)
        assert "OLED" in specs["AORUS MASTER 16 BZH"]["Display"]
