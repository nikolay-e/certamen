"""Shared test helpers and utilities for integration tests.

This module provides reusable functions and fixtures to reduce code
duplication across the test suite.
"""

from typing import Any

from arbitrium_core import Arbitrium
from tests.integration.conftest import MockModel

# ==================== Common Setup Helpers ====================


async def create_arbitrium_with_models(
    config: dict[str, Any],
    models: dict[str, MockModel] | None = None,
    skip_comparison: bool = False,
) -> Arbitrium:
    """Create an Arbitrium instance with mock models.

    Args:
        config: Configuration dictionary
        models: Optional dict of MockModel instances. If None, creates default 3 models
        skip_comparison: If True, skip creating the comparison object

    Returns:
        Configured Arbitrium instance
    """
    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
    )

    if models is None:
        models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "model_c": MockModel(model_name="test-c", display_name="Model C"),
        }

    arbitrium._healthy_models = models

    if not skip_comparison:
        arbitrium._comparison = arbitrium._create_comparison()

    return arbitrium


def add_judge_to_config(
    config: dict[str, Any], judge_key: str = "judge_model"
) -> None:
    """Add judge model configuration to config dict (modifies in place).

    Args:
        config: Configuration dictionary to modify
        judge_key: Key for the judge model
    """
    config["features"]["judge_model"] = judge_key
    config["models"][judge_key] = {
        "provider": "mock",
        "model_name": "judge",
        "display_name": "Judge Model",
        "temperature": 0.7,
        "max_tokens": 2000,
        "context_window": 8000,
    }


def create_model_with_size(
    name: str,
    size: str = "medium",
    **kwargs: Any,
) -> MockModel:
    """Create a MockModel with specific size characteristics.

    Args:
        name: Model name
        size: "small", "medium", or "large"
        **kwargs: Additional MockModel parameters

    Returns:
        Configured MockModel
    """
    size_configs = {
        "small": {"context_window": 2000, "max_tokens": 500},
        "medium": {"context_window": 4000, "max_tokens": 1000},
        "large": {"context_window": 128000, "max_tokens": 4096},
    }

    config = size_configs.get(size, size_configs["medium"])
    return MockModel(
        model_name=name,
        display_name=f"{name.title()} Model",
        **config,
        **kwargs,
    )


# ==================== Model Behavior Factories ====================


def create_failing_model(
    name: str, failure_condition: str = "always", **kwargs: Any
) -> MockModel:
    """Create a model that fails under specific conditions.

    Args:
        name: Model name
        failure_condition: When to fail - "always", "evaluation", "feedback", "after_n_calls"
        **kwargs: Additional parameters (e.g., call_threshold for "after_n_calls")

    Returns:
        MockModel configured to fail
    """
    if failure_condition == "always":
        return MockModel(
            model_name=name,
            display_name=f"{name.title()} Model",
            should_fail=True,
            **kwargs,
        )

    elif failure_condition == "evaluation":
        # Model that returns apology during evaluation
        return MockModel(
            model_name=name,
            display_name=f"{name.title()} Model",
            response_text="Sorry, I cannot evaluate these models.",
            **kwargs,
        )

    elif failure_condition == "feedback":
        # Model that fails during feedback generation
        return MockModel(
            model_name=name,
            display_name=f"{name.title()} Model",
            response_text="I cannot provide feedback.",
            **kwargs,
        )

    else:
        return MockModel(
            model_name=name,
            display_name=f"{name.title()} Model",
            **kwargs,
        )


# ==================== Common Assertions ====================


def assert_tournament_completed(result: str, metrics: dict[str, Any]) -> None:
    """Assert that a tournament completed successfully.

    Args:
        result: Tournament result string
        metrics: Tournament metrics dictionary
    """
    assert result is not None, "Tournament should return a result"
    assert len(result) > 0, "Tournament result should not be empty"
    assert "champion_model" in metrics, "Metrics should include champion_model"
    assert metrics["champion_model"] is not None, "Should have a champion"


def assert_models_eliminated(
    metrics: dict[str, Any],
    initial_count: int,
    expected_eliminated: int | None = None,
) -> None:
    """Assert that models were eliminated correctly.

    Args:
        metrics: Tournament metrics
        initial_count: Initial number of models
        expected_eliminated: Expected number of eliminations (if None, asserts > 0)
    """
    assert "eliminated_models" in metrics
    eliminated_count = len(metrics["eliminated_models"])

    if expected_eliminated is not None:
        assert (
            eliminated_count == expected_eliminated
        ), f"Expected {expected_eliminated} eliminations, got {eliminated_count}"
    else:
        assert eliminated_count > 0, "Should have at least one elimination"
        assert (
            eliminated_count < initial_count
        ), "Should not eliminate all models"


def assert_comparison_state(
    comparison: Any,
    active_models: int | None = None,
    has_judge: bool | None = None,
    has_kb: bool | None = None,
) -> None:
    """Assert comparison object state.

    Args:
        comparison: ModelComparison instance
        active_models: Expected number of active models
        has_judge: Whether judge model should be present
        has_kb: Whether knowledge bank should be enabled
    """
    if active_models is not None:
        assert len(comparison.active_model_keys) == active_models, (
            f"Expected {active_models} active models, "
            f"got {len(comparison.active_model_keys)}"
        )

    if has_judge is not None:
        if has_judge:
            assert (
                comparison.judge_model_key is not None
            ), "Should have judge model"
        else:
            assert (
                comparison.judge_model_key is None
            ), "Should not have judge model"

    if has_kb is not None:
        if has_kb:
            assert (
                comparison.knowledge_bank is not None
            ), "Should have knowledge bank"
        else:
            assert (
                comparison.knowledge_bank is None
                or not comparison.knowledge_bank.enabled
            ), "Should not have knowledge bank"
