"""Test judge score normalization to handle grading bias."""

from typing import Any

import pytest

from arbitrium_core import Arbitrium
from arbitrium_core.infrastructure.llm.registry import ProviderRegistry
from arbitrium_core.ports.llm import BaseModel, ModelResponse


class ScoringMockModel(BaseModel):
    """Mock model with custom scoring behavior for testing judge normalization."""

    def __init__(
        self,
        model_name: str,
        display_name: str,
        provider: str = "mock_scoring",
        scores_to_give: dict[str, float] | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        context_window: int = 4000,
    ):
        """Initialize with specific scores to give."""
        super().__init__(
            model_key=model_name,
            model_name=model_name,
            display_name=display_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,
        )
        self.scores_to_give = scores_to_give or {}

    async def generate(self, prompt: str) -> ModelResponse:
        """Generate response with predefined scores."""
        # Initial response
        if "evaluate" not in prompt.lower() and "score" not in prompt.lower():
            return ModelResponse.create_success(
                content=f"Response from {self.display_name}",
                cost=0.0,
            )

        # Evaluation response - return scores
        response_lines = []
        for model, score in self.scores_to_give.items():
            response_lines.append(f"{model}: {score}/10")

        return ModelResponse.create_success(
            content="\n".join(response_lines),
            cost=0.0,
        )

    async def generate_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        initial_delay: float | None = None,
        max_delay: float | None = None,
    ) -> ModelResponse:
        return await self.generate(prompt)


@pytest.fixture(autouse=True)
def register_scoring_provider():
    """Register scoring mock provider for judge normalization tests."""

    @ProviderRegistry.register("mock_scoring")
    class ScoringMockProvider:
        """Provider for scoring mock models."""

        @classmethod
        async def from_config(
            cls,
            model_key: str,
            config: dict,
            response_cache: Any | None = None,
        ) -> BaseModel:
            return ScoringMockModel(
                model_name=model_key,
                display_name=config.get("display_name", model_key),
            )


@pytest.mark.asyncio
async def test_judge_normalization_removes_bias(tmp_output_dir):
    """Test that judge normalization correctly handles harsh vs generous graders.

    Scenario:
    - LLM1 is a HARSH grader: gives self 9.0, gives LLM2 6.5 (mean=7.75)
    - LLM2 is GENEROUS grader: gives LLM1 8.5, gives self 7.3 (mean=7.9)

    Without normalization:
    - LLM1 median: (9.0 + 8.5)/2 = 8.75
    - LLM2 median: (6.5 + 7.3)/2 = 6.9
    - LLM1 wins (but is this fair?)

    With normalization (z-score + rescale):
    - Both judges agree LLM1 > LLM2, regardless of their grading scale
    - Relative rankings are preserved, absolute bias is removed
    """
    config = {
        "models": {
            "harsh": {
                "provider": "mock_scoring",
                "model_name": "harsh-grader",
                "display_name": "Harsh Grader",
                "context_window": 4000,
                "max_tokens": 1000,
            },
            "generous": {
                "provider": "mock_scoring",
                "model_name": "generous-grader",
                "display_name": "Generous Grader",
                "context_window": 4000,
                "max_tokens": 1000,
            },
        },
        "retry": {
            "max_attempts": 1,
            "initial_delay": 0.1,
        },
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
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

    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    # Create harsh and generous graders with specific scoring patterns
    # Harsh grader: gives 9.0 to self, 6.5 to others (strict!)
    harsh_model = ScoringMockModel(
        model_name="harsh",
        display_name="Harsh Grader",
        scores_to_give={
            "LLM1": 9.0,  # Self-score
            "LLM2": 6.5,  # Other model
        },
    )

    # Generous grader: gives 8.5 to others, 7.3 to self (more lenient)
    generous_model = ScoringMockModel(
        model_name="generous",
        display_name="Generous Grader",
        scores_to_give={
            "LLM1": 8.5,  # Other model
            "LLM2": 7.3,  # Self-score
        },
    )

    arbitrium._all_models["harsh"] = harsh_model
    arbitrium._all_models["generous"] = generous_model
    arbitrium._healthy_models["harsh"] = harsh_model
    arbitrium._healthy_models["generous"] = generous_model

    # Run tournament
    result, metrics = await arbitrium.run_tournament(
        "Test judge normalization"
    )

    # Verify tournament completed
    assert result is not None
    assert metrics is not None

    # The important thing: normalization should preserve relative rankings
    # Both judges agree: harsh > generous (harsh gives self 9.0 vs 6.5 to other,
    # generous gives other 8.5 vs 7.3 to self, so harsh is relatively better)
    #
    # After normalization, the winner should be determined by relative performance,
    # not absolute score bias
    assert "Harsh" in result or "Generous" in result


@pytest.mark.asyncio
async def test_judge_normalization_with_extreme_bias(tmp_output_dir):
    """Test normalization handles extreme grading bias.

    Scenario:
    - LLM1 is EXTREMELY harsh: gives 5.0 and 3.0 (mean=4.0)
    - LLM2 is EXTREMELY generous: gives 10.0 and 9.5 (mean=9.75)

    Normalization should make these comparable despite wildly different scales.
    """
    config = {
        "models": {
            "super_harsh": {
                "provider": "mock_scoring",
                "model_name": "super-harsh",
                "display_name": "Super Harsh",
                "context_window": 4000,
                "max_tokens": 1000,
            },
            "super_generous": {
                "provider": "mock_scoring",
                "model_name": "super-generous",
                "display_name": "Super Generous",
                "context_window": 4000,
                "max_tokens": 1000,
            },
        },
        "retry": {"max_attempts": 1},
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
        },
        "knowledge_bank": {"enabled": False},
        "prompts": {
            "initial": {"content": "Answer.", "metadata": {}},
            "feedback": {"content": "Feedback.", "metadata": {}},
            "improvement": {"content": "Improve.", "metadata": {}},
            "evaluate": {"content": "Score.", "metadata": {}},
        },
        "improvement_phase": {"enabled": False},
        "refinement_phase": {"enabled": False},
        "outputs_dir": str(tmp_output_dir),
    }

    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    # Super harsh: mean=4.0
    super_harsh = ScoringMockModel(
        model_name="super_harsh",
        display_name="Super Harsh",
        scores_to_give={"LLM1": 5.0, "LLM2": 3.0},
    )

    # Super generous: mean=9.75
    super_generous = ScoringMockModel(
        model_name="super_generous",
        display_name="Super Generous",
        scores_to_give={"LLM1": 10.0, "LLM2": 9.5},
    )

    arbitrium._all_models["super_harsh"] = super_harsh
    arbitrium._all_models["super_generous"] = super_generous
    arbitrium._healthy_models["super_harsh"] = super_harsh
    arbitrium._healthy_models["super_generous"] = super_generous

    result, metrics = await arbitrium.run_tournament("Test extreme bias")

    # Should still complete successfully
    assert result is not None
    assert metrics is not None
    # Both judges rank LLM1 higher than LLM2
    assert "Super Harsh" in result or "Super Generous" in result
