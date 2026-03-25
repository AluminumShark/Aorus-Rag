"""Shared test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_specs() -> dict[str, dict[str, str]]:
    """Minimal 3-model specs dict with shared + differing categories."""
    return {
        "AORUS MASTER 16 BZH": {
            "OS": "Windows 11 Pro\nWindows 11 Home",
            "Video Graphics": "NVIDIA RTX 5090\n24GB GDDR7",
        },
        "AORUS MASTER 16 BYH": {
            "OS": "Windows 11 Pro\nWindows 11 Home",
            "Video Graphics": "NVIDIA RTX 5080\n16GB GDDR7",
        },
        "AORUS MASTER 16 BXH": {
            "OS": "Windows 11 Pro\nWindows 11 Home",
            "Video Graphics": "NVIDIA RTX 5070 Ti\n12GB GDDR7",
        },
    }
