"""Unit tests for src.pipeline.query — mock all heavy components."""

from __future__ import annotations

from unittest.mock import patch

from src.pipeline import query, _detect_lang


class TestQuery:
    """Test pipeline query logic without loading any models."""

    def _make_pipe(self, chunks: list[dict]) -> dict:
        """Create a minimal pipeline dict with fake chunks."""
        return {"model": None, "index": None, "chunks": chunks, "llm": None}

    @patch("src.pipeline.generate", return_value=iter(["token1", "token2"]))
    @patch(
        "src.pipeline.search", return_value=([0.8, 0.5, 0.4, 0.3, 0.2], [0, 1, 2, 3, 4])
    )
    @patch("src.pipeline.encode", return_value=[[0.1] * 384])
    def test_returns_iterator(self, mock_encode, mock_search, mock_generate) -> None:
        chunks = [{"text": f"chunk{i}"} for i in range(5)]
        pipe = self._make_pipe(chunks)
        tokens = list(query(pipe, "test query"))
        assert tokens == ["token1", "token2"]

    @patch("src.pipeline.generate", return_value=iter(["ans"]))
    @patch("src.pipeline.search", return_value=([0.8], [0]))
    @patch("src.pipeline.encode", return_value=[[0.1] * 384])
    def test_encode_called_with_is_query(
        self, mock_encode, mock_search, mock_generate
    ) -> None:
        """Pipeline must pass is_query=True when encoding user queries."""
        pipe = self._make_pipe([{"text": "chunk0"}])
        list(query(pipe, "test"))
        mock_encode.assert_called_once_with(None, ["test"], is_query=True)

    @patch("src.pipeline.generate", return_value=iter(["ans"]))
    @patch(
        "src.pipeline.search", return_value=([0.9, 0.7, 0.5, 0.4, 0.3], [2, 0, 1, 3, 4])
    )
    @patch("src.pipeline.encode", return_value=[[0.1] * 384])
    def test_context_joins_correct_chunks(
        self, mock_encode, mock_search, mock_generate
    ) -> None:
        chunks = [
            {"text": "chunk-A"},
            {"text": "chunk-B"},
            {"text": "chunk-C"},
            {"text": "chunk-D"},
            {"text": "chunk-E"},
        ]
        pipe = self._make_pipe(chunks)
        list(query(pipe, "test"))  # consume iterator
        call_args = mock_generate.call_args
        context = call_args[0][1]  # second positional arg
        assert context == "chunk-C\nchunk-A\nchunk-B\nchunk-D\nchunk-E"
        # "test" is ASCII → lang should be "en"
        assert call_args[1]["lang"] == "en"

    @patch("src.pipeline.generate", return_value=iter(["response"]))
    @patch(
        "src.pipeline.search", return_value=([0.5, 0.4, 0.3, 0.2, 0.1], [0, 1, 2, 3, 4])
    )
    @patch("src.pipeline.encode", return_value=[[0.1] * 384])
    def test_empty_query_still_works(
        self, mock_encode, mock_search, mock_generate
    ) -> None:
        """Empty string query should not crash the pipeline."""
        chunks = [{"text": f"c{i}"} for i in range(5)]
        pipe = self._make_pipe(chunks)
        tokens = list(query(pipe, ""))
        assert tokens == ["response"]

    @patch("src.pipeline.generate", return_value=iter(["ans"]))
    @patch("src.pipeline.search", return_value=([0.8, 0.5, 0.3], [0, 1, 2]))
    @patch("src.pipeline.encode", return_value=[[0.1] * 384])
    def test_context_from_partial_results(
        self, mock_encode, mock_search, mock_generate
    ) -> None:
        """Search returns fewer results than top_k (3 instead of 5)."""
        chunks = [{"text": "A"}, {"text": "B"}, {"text": "C"}]
        pipe = self._make_pipe(chunks)
        list(query(pipe, "test"))
        context = mock_generate.call_args[0][1]
        assert context == "A\nB\nC"

    @patch("src.pipeline.generate", return_value=iter(["答案"]))
    @patch("src.pipeline.search", return_value=([0.1, 0.05], [0, 1]))
    @patch("src.pipeline.encode", return_value=[[0.1] * 384])
    def test_low_score_still_generates(
        self, mock_encode, mock_search, mock_generate
    ) -> None:
        """Even low-scoring results are passed to LLM (no threshold)."""
        chunks = [{"text": "c0"}, {"text": "c1"}]
        pipe = self._make_pipe(chunks)
        tokens = list(query(pipe, "隨便問"))
        assert tokens == ["答案"]


class TestLanguageDetection:
    """Test _detect_lang."""

    def test_english_query(self) -> None:
        assert _detect_lang("What is the weather today?") == "en"

    def test_chinese_query(self) -> None:
        assert _detect_lang("這台筆電多少錢？") == "zh"

    def test_mixed_mostly_chinese(self) -> None:
        assert _detect_lang("處理器是什麼型號？") == "zh"

    def test_mixed_cjk_and_english(self) -> None:
        assert _detect_lang("我的central process unit是甚麼型號") == "zh"

    def test_empty_string(self) -> None:
        assert _detect_lang("") == "en"  # no CJK chars → en
