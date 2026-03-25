"""Unit tests for src.generator — think tag filtering logic.

Mocks llm.create_chat_completion to test the filtering without loading a model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.generator import generate


def _mock_llm(tokens: list[str]) -> MagicMock:
    """Create a mock LLM that streams chat completion chunks."""
    llm = MagicMock()
    llm.create_chat_completion.return_value = iter(
        {"choices": [{"delta": {"content": t}}]} for t in tokens
    )
    return llm


class TestThinkTagFiltering:
    def test_filters_think_block(self) -> None:
        llm = _mock_llm(["<think>", "reasoning", "</think>", "answer"])
        result = list(generate(llm, "context", "query"))
        assert result == ["answer"]

    def test_normal_tokens_pass_through(self) -> None:
        llm = _mock_llm(["Hello", " world"])
        result = list(generate(llm, "context", "query"))
        assert result == ["Hello", " world"]

    def test_unclosed_think_eats_all(self) -> None:
        """If <think> is never closed, all subsequent tokens are swallowed."""
        llm = _mock_llm(["<think>", "a", "b"])
        result = list(generate(llm, "context", "query"))
        assert result == []

    def test_leading_newlines_pass_through(self) -> None:
        """All tokens pass through (no stripping or stop logic)."""
        llm = _mock_llm(["\n\nHello"])
        result = list(generate(llm, "context", "query"))
        assert result == ["\n\nHello"]

    def test_whitespace_tokens_pass_through(self) -> None:
        """Whitespace tokens are not filtered."""
        llm = _mock_llm(["\n", "\n", "Hello", " world"])
        result = list(generate(llm, "context", "query"))
        assert result == ["\n", "\n", "Hello", " world"]

    def test_think_then_newlines_then_content(self) -> None:
        """Think block filtered, everything else passes through."""
        llm = _mock_llm(["<think>", "hmm", "</think>", "\n\n", "answer"])
        result = list(generate(llm, "context", "query"))
        assert result == ["\n\n", "answer"]

    @pytest.mark.xfail(
        reason="Known bug: <think> and </think> in same token — "
               "continue on <think> match skips closing tag check",
        strict=True,
    )
    def test_think_open_close_same_token(self) -> None:
        """If a single token contains both <think> and </think>,
        the current code hits <think> first, continues, and never
        processes the closing tag — leaving in_think=True forever."""
        llm = _mock_llm(["<think>blah</think>", "visible"])
        result = list(generate(llm, "context", "query"))
        assert result == ["visible"]
