"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()  # type: ignore[misc]
def tmp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()  # type: ignore[misc]
def sample_config() -> dict[str, Any]:
    """Sample configuration for testing."""
    return {
        "models": {
            "test_model": {
                "provider": "ollama",
                "model_name": "ollama/qwen3:8b",
                "display_name": "Test Model",
                "temperature": 0.7,
            }
        },
        "retry": {
            "max_attempts": 2,
            "initial_delay": 1,
            "max_delay": 5,
        },
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
            "judge_model": None,
            "knowledge_bank_model": "leader",
            "llm_compression": False,
        },
        "knowledge_bank": {
            "enabled": False,
            "similarity_threshold": 0.75,
            "max_insights_to_inject": 5,
        },
        "prompts": {
            "initial": {
                "content": "Test initial prompt",
                "metadata": {
                    "version": "1.0",
                    "type": "instruction",
                    "phase": "initial_response",
                },
            },
            "feedback": {
                "content": "Test feedback prompt",
                "metadata": {
                    "version": "1.0",
                    "type": "instruction",
                    "phase": "feedback",
                },
            },
            "improvement": {
                "content": "Test improvement prompt",
                "metadata": {
                    "version": "1.0",
                    "type": "instruction",
                    "phase": "improvement",
                },
            },
            "evaluate": {
                "content": "Test evaluate prompt",
                "metadata": {
                    "version": "1.0",
                    "type": "instruction",
                    "phase": "evaluation",
                },
            },
        },
    }


@pytest.fixture()  # type: ignore[misc]
def sample_question() -> str:
    """Sample question for testing."""
    return "What is the capital of France?"
