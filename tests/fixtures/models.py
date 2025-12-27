from typing import Any


def create_mock_model_config(
    name: str = "test-model-1", provider: str = "mock", **overrides
) -> dict[str, Any]:
    """Factory for creating mock model configurations.

    Args:
        name: Model name (e.g., "test-model-1", "gpt-4o")
        provider: Provider name (default: "mock")
        **overrides: Additional fields to override defaults

    Returns:
        Dict with mock model configuration

    Example:
        >>> create_mock_model_config("test-claude")
        {'provider': 'mock', 'model_name': 'test-claude', ...}

        >>> create_mock_model_config("gpt-4o", context_window=8000)
        {'provider': 'mock', 'model_name': 'gpt-4o', 'context_window': 8000, ...}
    """
    config = {
        "provider": provider,
        "model_name": name,
        "display_name": name.replace("test-", "Model ")
        .replace("-", " ")
        .title(),
        "context_window": 4000,
        "max_tokens": 1000,
        "temperature": 0.7,
    }
    config.update(overrides)
    return config


STANDARD_TEST_PROMPTS = {
    "initial": {
        "content": "Test initial prompt content",
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "initial_response",
        },
    },
    "feedback": {
        "content": "Test feedback prompt content",
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "feedback",
        },
    },
    "improvement": {
        "content": "Test improvement prompt content",
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "improvement",
        },
    },
    "evaluate": {
        "content": "Test evaluate prompt content",
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "evaluation",
        },
    },
}
