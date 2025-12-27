"""Integration test fixtures and utilities."""

import asyncio
import re
import tempfile
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from arbitrium_core import Arbitrium
from arbitrium_core.infrastructure.llm.registry import ProviderRegistry
from arbitrium_core.ports.llm import BaseModel, ModelResponse


def pytest_addoption(parser):
    parser.addoption(
        "--run-real-llm",
        action="store_true",
        default=False,
        help="Run tests that make actual LLM API calls (requires ollama)",
    )


class MockModel(BaseModel):
    """Mock model for integration testing."""

    def __init__(
        self,
        model_name: str = "test-model",
        display_name: str = "Test Model",
        provider: str = "mock",
        response_text: str = "This is a comprehensive mock response with sufficient detail for knowledge bank validation",
        should_fail: bool = False,
        delay: float = 0.0,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        context_window: int = 4000,
    ):
        """Initialize mock model."""
        # Initialize parent BaseModel with required parameters
        super().__init__(
            model_key=model_name,
            model_name=model_name,
            display_name=display_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,
        )

        # Mock-specific attributes
        self._response_text = response_text
        self._should_fail = should_fail
        self._delay = delay
        self._call_count = 0

    async def generate(self, prompt: str) -> ModelResponse:
        """Generate mock response."""
        self._call_count += 1

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._should_fail:
            return ModelResponse.create_error(
                "Mock failure", provider=self.provider
            )

        # Vary responses based on call count for more realistic testing
        if "evaluate" in prompt.lower() or "score" in prompt.lower():
            # If response_text was explicitly set to something different from default,
            # and it looks like apology/refusal, use it instead of auto-generating scores
            default_response = "This is a comprehensive mock response with sufficient detail for knowledge bank validation"
            if self._response_text != default_response and any(
                keyword in self._response_text.lower()
                for keyword in [
                    "sorry",
                    "cannot",
                    "can't",
                    "unable",
                    "apologize",
                ]
            ):
                # Use the custom response text (likely an apology/refusal)
                response = self._response_text
            else:
                # Extract model names from the prompt to provide adaptive scores
                # Look for patterns like "LLM1:", "Model A:", etc.
                model_names = re.findall(
                    r"(LLM\d+|Model [A-Z]|Response \d+)", prompt
                )
                unique_models = list(
                    dict.fromkeys(model_names)
                )  # Preserve order, remove duplicates

                if unique_models:
                    # Generate scores for the actual models in the prompt
                    scores = []
                    for i, model in enumerate(unique_models):
                        # Vary scores to ensure there's a winner and loser
                        score = (
                            8 - i
                        )  # First model gets 8, second gets 7, etc.
                        scores.append(f"{model}: {score}/10")
                    response = "\n".join(scores)
                else:
                    # Fallback to generic response if no models detected
                    response = "Model A: 8/10\nModel B: 7/10"
        elif "improve" in prompt.lower() or "refine" in prompt.lower():
            response = (
                f"Improved: {self._response_text} (call {self._call_count})"
            )
        elif "feedback" in prompt.lower():
            response = "Feedback: This answer could be improved by adding more detail."
        elif "extract" in prompt.lower() or "insight" in prompt.lower():
            # Provide longer insights to pass the 50-character minimum validation
            response = (
                "- The primary consideration here is the long-term sustainability of the approach\n"
                "- Historical precedent suggests this strategy has proven effective in similar contexts\n"
                "- Cost-benefit analysis indicates significant potential for optimization"
            )
        else:
            # Only add call tracking suffix if response_text is non-empty
            if self._response_text:
                response = f"{self._response_text} (call {self._call_count})"
            else:
                response = self._response_text

        return ModelResponse(
            content=response,
            cost=0.001,
            provider=self.provider,
        )

    async def generate_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        initial_delay: float | None = None,
        max_delay: float | None = None,
    ) -> ModelResponse:
        """Mock implementation that just calls generate without retry."""
        return await self.generate(prompt)


# Register MockModel as a provider for testing
@ProviderRegistry.register("mock")
class MockProvider:
    """Provider for mock models in tests."""

    @classmethod
    async def from_config(
        cls,
        model_key: str,
        config: dict[str, Any],
        response_cache: Any | None = None,
    ) -> BaseModel:
        return MockModel(
            model_name=str(config.get("model_name", model_key)),
            display_name=str(config.get("display_name", model_key)),
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=int(config.get("max_tokens", 1000)),
            context_window=int(config.get("context_window", 4000)),
        )


def make_model_config(
    name: str,
    display_name: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return {
        "provider": kwargs.get("provider", "mock"),
        "model_name": kwargs.get("model_name", f"test-{name}"),
        "display_name": display_name or f"Model {name.upper()}",
        "context_window": kwargs.get("context_window", 4000),
        "max_tokens": kwargs.get("max_tokens", 1000),
        "temperature": kwargs.get("temperature", 0.7),
    }


def make_mock_model(
    name: str,
    display_name: str | None = None,
    response_text: str | None = None,
    should_fail: bool = False,
    **kwargs: Any,
) -> MockModel:
    display = display_name or f"Model {name.upper()}"
    text = response_text or f"Model {name.upper()}'s response"
    return MockModel(
        model_name=f"test-{name}",
        display_name=display,
        response_text=text,
        should_fail=should_fail,
        **kwargs,
    )


def make_mock_models_dict(
    names: list[str] | None = None,
    responses: dict[str, str] | None = None,
) -> dict[str, MockModel]:
    names = names or ["a", "b", "c"]
    responses = responses or {}
    return {
        f"model_{name}": make_mock_model(
            name=name,
            response_text=responses.get(name),
        )
        for name in names
    }


def make_prompt(content: str, phase: str) -> dict[str, Any]:
    return {
        "content": content,
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": phase,
        },
    }


def make_prompts(
    initial: str = "Answer the question.",
    feedback: str = "Provide feedback.",
    improvement: str = "Improve the answer.",
    evaluate: str = "Score the responses.",
) -> dict[str, dict[str, Any]]:
    return {
        "initial": make_prompt(initial, "initial_response"),
        "feedback": make_prompt(feedback, "feedback"),
        "improvement": make_prompt(improvement, "improvement"),
        "evaluate": make_prompt(evaluate, "evaluation"),
    }


@pytest.fixture()
def tmp_output_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def basic_config(tmp_output_dir: Path) -> dict[str, Any]:
    return {
        "models": {
            "model_a": make_model_config("a"),
            "model_b": make_model_config("b"),
            "model_c": make_model_config("c"),
        },
        "retry": {
            "max_attempts": 2,
            "initial_delay": 0.1,
            "max_delay": 1,
        },
        "features": {
            "save_reports_to_disk": True,
            "deterministic_mode": True,
            "judge_model": None,
            "knowledge_bank_model": "leader",
            "llm_compression": False,
        },
        "knowledge_bank": {
            "enabled": False,
            "similarity_threshold": 0.75,
            "max_insights": 100,
        },
        "prompts": make_prompts(
            initial="Please answer the following question clearly and concisely.",
            feedback="Provide constructive feedback on the answer.",
            improvement="Improve your answer based on the context provided.",
            evaluate="Evaluate the responses and provide scores.",
        ),
        "improvement_phase": {
            "enabled": True,
            "feedback_enabled": False,
            "share_responses": True,
        },
        "refinement_phase": {
            "enabled": True,
            "feedback_enabled": False,
            "share_responses": True,
        },
        "outputs_dir": str(tmp_output_dir),
    }


@pytest.fixture()
def kb_enabled_config(basic_config: dict[str, Any]) -> dict[str, Any]:
    """Configuration with knowledge bank enabled."""
    config = basic_config.copy()
    config["knowledge_bank"]["enabled"] = True
    return config


@pytest.fixture()
def minimal_config(tmp_output_dir: Path) -> dict[str, Any]:
    return {
        "models": {
            "model_a": make_model_config("a"),
            "model_b": make_model_config("b"),
        },
        "retry": {
            "max_attempts": 2,
            "initial_delay": 0.1,
            "max_delay": 1,
        },
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
            "judge_model": None,
            "llm_compression": False,
        },
        "knowledge_bank": {
            "enabled": False,
        },
        "prompts": make_prompts(),
        "improvement_phase": {
            "enabled": False,
        },
        "refinement_phase": {
            "enabled": False,
        },
        "outputs_dir": str(tmp_output_dir),
    }


@pytest.fixture()
def mock_models() -> dict[str, MockModel]:
    """Create mock models for testing."""
    return {
        "model_a": MockModel(
            model_name="test-a",
            display_name="Model A",
            response_text="Model A's detailed answer",
        ),
        "model_b": MockModel(
            model_name="test-b",
            display_name="Model B",
            response_text="Model B's comprehensive response",
        ),
        "model_c": MockModel(
            model_name="test-c",
            display_name="Model C",
            response_text="Model C's thorough analysis",
        ),
    }


@pytest.fixture()
def failing_model() -> MockModel:
    """Create a mock model that always fails."""
    return MockModel(
        model_name="failing-model",
        display_name="Failing Model",
        should_fail=True,
    )


@pytest_asyncio.fixture()
async def arbitrium_instance(
    basic_config: dict[str, Any],
    mock_models: dict[str, MockModel],
) -> AsyncGenerator[Arbitrium, None]:
    """Create an Arbitrium instance with mock models."""
    # Skip health check to avoid LiteLLM calls
    arbitrium = await Arbitrium.from_settings(
        settings=basic_config,
        skip_secrets=True,
        skip_health_check=True,
    )

    # Replace all models with mocks (both all_models and healthy_models)
    arbitrium._all_models = mock_models  # type: ignore[assignment]
    arbitrium._healthy_models = mock_models  # type: ignore[assignment]

    yield arbitrium


@pytest.fixture()
def simple_question() -> str:
    """Simple test question."""
    return "What is the capital of France?"


@pytest.fixture()
def complex_question() -> str:
    """Complex test question requiring analysis."""
    return """
    Should a startup with 50 employees migrate from monolithic architecture
    to microservices? Consider technical feasibility, team capacity, and
    business impact.
    """.strip()


@pytest.fixture()
def test_questions() -> list[str]:
    """Multiple test questions for batch testing."""
    return [
        "What is 2+2?",
        "Explain quantum computing in simple terms.",
        "What are the benefits of remote work?",
    ]
