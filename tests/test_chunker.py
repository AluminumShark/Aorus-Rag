"""Unit tests for src.chunker — pure logic, no model loading."""

from __future__ import annotations

import json

import pytest

from src.chunker import _clean_value, create_chunks


class TestCleanValue:
    def test_basic_multiline(self) -> None:
        assert _clean_value("line1\nline2\nline3") == "line1, line2, line3"

    def test_strips_whitespace(self) -> None:
        assert _clean_value("  a \n b  \n c ") == "a, b, c"

    def test_single_line(self) -> None:
        assert _clean_value("hello") == "hello"

    def test_blank_lines_skipped(self) -> None:
        assert _clean_value("a\n\n\nb") == "a, b"

    def test_empty_string(self) -> None:
        assert _clean_value("") == ""

    def test_whitespace_only_value(self) -> None:
        assert _clean_value("\n  \n  \n") == ""


class TestCreateChunks:
    def test_shared_spec_produces_one_chunk(self, sample_specs: dict) -> None:
        chunks = create_chunks(sample_specs)
        os_chunks = [c for c in chunks if c["category"] == "OS"]
        assert len(os_chunks) == 1
        assert "(shared by BZH/BYH/BXH)" in os_chunks[0]["text"]
        assert "model" not in os_chunks[0]  # shared chunks have no model key

    def test_shared_chunk_has_bilingual_alias(self, sample_specs: dict) -> None:
        chunks = create_chunks(sample_specs)
        os_chunks = [c for c in chunks if c["category"] == "OS"]
        assert "(作業系統)" in os_chunks[0]["text"]

    def test_differing_spec_produces_per_model_chunks(self, sample_specs: dict) -> None:
        chunks = create_chunks(sample_specs)
        gpu_chunks = [c for c in chunks if c["category"] == "Video Graphics"]
        assert len(gpu_chunks) == 3
        models = {c["model"] for c in gpu_chunks}
        assert models == {
            "AORUS MASTER 16 BZH",
            "AORUS MASTER 16 BYH",
            "AORUS MASTER 16 BXH",
        }
        assert "[AORUS MASTER 16 BZH]" in gpu_chunks[0]["text"]

    def test_per_model_chunk_has_bilingual_alias(self, sample_specs: dict) -> None:
        chunks = create_chunks(sample_specs)
        gpu_chunks = [c for c in chunks if c["category"] == "Video Graphics"]
        for c in gpu_chunks:
            assert "(顯示卡/顯卡/GPU)" in c["text"]

    def test_mixed_total_count(self, sample_specs: dict) -> None:
        chunks = create_chunks(sample_specs)
        # 1 shared (OS) + 3 per-model (Video Graphics) = 4
        assert len(chunks) == 4

    def test_clean_value_applied_in_text(self, sample_specs: dict) -> None:
        chunks = create_chunks(sample_specs)
        for c in chunks:
            assert "\n" not in c["text"]

    def test_with_real_specs(self) -> None:
        """Validate against the actual specs.json on disk."""
        path = "data/raw/specs.json"
        try:
            with open(path, encoding="utf-8") as f:
                specs = json.load(f)
        except FileNotFoundError:
            pytest.skip("data/raw/specs.json not found (run build_index first)")
        chunks = create_chunks(specs)
        assert len(chunks) == 19
