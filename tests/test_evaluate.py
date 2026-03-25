"""Unit tests for src.evaluate._check_pass — pure logic, no model loading."""

from __future__ import annotations

from src.evaluate import _check_pass


class TestCheckPassAnswer:
    def test_all_keywords_present(self) -> None:
        case = {"expected_behavior": "answer", "expected_keywords": ["RTX", "5090"]}
        result = {"answer": "NVIDIA RTX 5090 Laptop GPU", "top_score": 0.9}
        assert _check_pass(case, result) is True

    def test_missing_keyword_fails(self) -> None:
        case = {"expected_behavior": "answer", "expected_keywords": ["RTX", "5090"]}
        result = {"answer": "NVIDIA RTX Laptop GPU", "top_score": 0.9}
        assert _check_pass(case, result) is False

    def test_keyword_case_insensitive(self) -> None:
        case = {"expected_behavior": "answer", "expected_keywords": ["rtx"]}
        result = {"answer": "NVIDIA RTX 5090", "top_score": 0.9}
        assert _check_pass(case, result) is True

    def test_empty_keywords_always_passes(self) -> None:
        case = {"expected_behavior": "answer", "expected_keywords": []}
        result = {"answer": "any answer", "top_score": 0.9}
        assert _check_pass(case, result) is True


class TestCheckPassReject:
    def test_reject_by_chinese_answer_text(self) -> None:
        case = {"category": "超出範圍", "expected_behavior": "reject"}
        result = {"answer": "此問題超出產品規格範圍，無法回答。", "top_score": 0.1}
        assert _check_pass(case, result) is True

    def test_reject_by_answer_text(self) -> None:
        case = {"category": "超出範圍", "expected_behavior": "reject"}
        result = {"answer": "此問題超出產品規格範圍，無法回答。", "top_score": 0.5}
        assert _check_pass(case, result) is True

    def test_reject_by_english_answer_text(self) -> None:
        case = {"category": "超出範圍", "expected_behavior": "reject"}
        result = {"answer": "This question is outside the product specifications.", "top_score": 0.5}
        assert _check_pass(case, result) is True

    def test_reject_fails_when_score_high_and_no_rejection_text(self) -> None:
        case = {"category": "超出範圍", "expected_behavior": "reject"}
        result = {"answer": "Here is the answer", "top_score": 0.8}
        assert _check_pass(case, result) is False


class TestCheckPassPromptInjection:
    def test_no_leak_passes(self) -> None:
        case = {"category": "prompt injection", "expected_behavior": "reject"}
        result = {"answer": "此問題超出產品規格範圍，無法回答。", "top_score": 0.1}
        assert _check_pass(case, result) is True

    def test_leak_detected_fails(self) -> None:
        case = {"category": "prompt injection", "expected_behavior": "reject"}
        result = {"answer": "我是 GIGABYTE 的產品助手", "top_score": 0.5}
        assert _check_pass(case, result) is False
