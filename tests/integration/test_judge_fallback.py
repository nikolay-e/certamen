"""Test judge fallback strategy."""

from typing import Any

import pytest

from arbitrium_core import Arbitrium
from arbitrium_core.infrastructure.llm.registry import ProviderRegistry
from arbitrium_core.ports.llm import BaseModel, ModelResponse


class FailingMockModel(BaseModel):
    """Mock model that always fails for testing fallback."""

    def __init__(
        self,
        model_name: str = "failing-judge",
        display_name: str = "Failing Judge",
        provider: str = "mock_failing",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        context_window: int = 4000,
    ):
        """Initialize failing mock model."""
        super().__init__(
            model_key=model_name,
            model_name=model_name,
            display_name=display_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,
        )

    async def generate(self, prompt: str) -> ModelResponse:
        """Always return an error."""
        return ModelResponse.create_error(
            "Simulated judge failure (rate limit)",
            error_type="rate_limit",
            provider=self.provider,
        )

    async def generate_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        initial_delay: float | None = None,
        max_delay: float | None = None,
    ) -> ModelResponse:
        """Mock retry that just calls generate."""
        return await self.generate(prompt)


@pytest.fixture(autouse=True)
def register_failing_provider():
    """Register failing mock provider for this test module."""

    @ProviderRegistry.register("mock_failing")
    class FailingMockProvider:
        """Provider for failing mock models."""

        @classmethod
        async def from_config(
            cls,
            model_key: str,
            config: dict,
            response_cache: Any | None = None,
        ) -> BaseModel:
            return FailingMockModel(
                model_name=str(config.get("model_name", model_key)),
                display_name=str(config.get("display_name", model_key)),
                max_tokens=int(config.get("max_tokens", 1000)),
                temperature=float(config.get("temperature", 0.7)),
                context_window=int(config.get("context_window", 4000)),
            )


@pytest.mark.asyncio
async def test_judge_fallback_to_emergency_judge(tmp_output_dir, mock_models):
    """Test that when primary judge fails, emergency judge is used."""
    # Create config with failing judge and working models
    config = {
        "models": {
            "model_a": {
                "provider": "mock",
                "model_name": "test-a",
                "display_name": "Model A",
                "context_window": 4000,
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            "model_b": {
                "provider": "mock",
                "model_name": "test-b",
                "display_name": "Model B",
                "context_window": 8000,  # Largest context - will be emergency judge
                "max_tokens": 2000,
                "temperature": 0.7,
            },
            "failing_judge": {
                "provider": "mock_failing",
                "model_name": "failing-judge",
                "display_name": "Failing Judge",
                "context_window": 16000,  # Largest, but will fail
                "max_tokens": 4000,
                "temperature": 0.7,
            },
        },
        "retry": {
            "max_attempts": 2,
            "initial_delay": 0.1,
            "max_delay": 1,
        },
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
            "judge_model": "failing_judge",  # Use failing judge
        },
        "knowledge_bank": {
            "enabled": False,
        },
        "prompts": {
            "initial": {
                "content": "Answer the question.",
                "metadata": {"version": "1.0", "type": "instruction"},
            },
            "feedback": {
                "content": "Provide feedback.",
                "metadata": {"version": "1.0", "type": "instruction"},
            },
            "improvement": {
                "content": "Improve the answer.",
                "metadata": {"version": "1.0", "type": "instruction"},
            },
            "evaluate": {
                "content": "Score the responses.",
                "metadata": {"version": "1.0", "type": "instruction"},
            },
        },
        "improvement_phase": {"enabled": False},
        "refinement_phase": {"enabled": False},
        "outputs_dir": str(tmp_output_dir),
    }

    # Create Arbitrium instance
    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    # Replace models with our mocks (keeping the failing judge)
    arbitrium._all_models["model_a"] = mock_models["model_a"]
    arbitrium._all_models["model_b"] = mock_models["model_b"]
    arbitrium._healthy_models["model_a"] = mock_models["model_a"]
    arbitrium._healthy_models["model_b"] = mock_models["model_b"]

    # Run tournament - should fallback when judge fails
    result, metrics = await arbitrium.run_tournament("What is 2+2?")

    # Tournament should complete successfully using fallback
    assert result is not None
    assert metrics is not None
    assert "Model A" in result or "Model B" in result


@pytest.mark.asyncio
async def test_judge_fallback_to_peer_review(tmp_output_dir, mock_models):
    """Test that when all judges fail, peer review is used as last resort."""
    # Create config with only failing judges
    config = {
        "models": {
            "model_a": {
                "provider": "mock",
                "model_name": "test-a",
                "display_name": "Model A",
                "context_window": 4000,
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            "model_b": {
                "provider": "mock",
                "model_name": "test-b",
                "display_name": "Model B",
                "context_window": 4000,
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            "failing_judge_1": {
                "provider": "mock_failing",
                "model_name": "failing-judge-1",
                "display_name": "Failing Judge 1",
                "context_window": 8000,
                "max_tokens": 2000,
                "temperature": 0.7,
            },
        },
        "retry": {
            "max_attempts": 2,
            "initial_delay": 0.1,
            "max_delay": 1,
        },
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
            "judge_model": "failing_judge_1",  # Use failing judge
        },
        "knowledge_bank": {
            "enabled": False,
        },
        "prompts": {
            "initial": {
                "content": "Answer the question.",
                "metadata": {"version": "1.0", "type": "instruction"},
            },
            "feedback": {
                "content": "Provide feedback.",
                "metadata": {"version": "1.0", "type": "instruction"},
            },
            "improvement": {
                "content": "Improve the answer.",
                "metadata": {"version": "1.0", "type": "instruction"},
            },
            "evaluate": {
                "content": "Score the responses.",
                "metadata": {"version": "1.0", "type": "instruction"},
            },
        },
        "improvement_phase": {"enabled": False},
        "refinement_phase": {"enabled": False},
        "outputs_dir": str(tmp_output_dir),
    }

    # Create Arbitrium instance
    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    # Replace participant models with working mocks
    arbitrium._all_models["model_a"] = mock_models["model_a"]
    arbitrium._all_models["model_b"] = mock_models["model_b"]
    arbitrium._healthy_models["model_a"] = mock_models["model_a"]
    arbitrium._healthy_models["model_b"] = mock_models["model_b"]

    # Run tournament - should fallback to peer review
    result, metrics = await arbitrium.run_tournament("What is 2+2?")

    # Tournament should complete successfully using peer review
    assert result is not None
    assert metrics is not None
    assert "Model A" in result or "Model B" in result
