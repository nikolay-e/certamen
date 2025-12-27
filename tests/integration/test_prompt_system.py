"""
Integration tests for the prompt system.

Tests the refactored prompt management:
- Prompt formatting with dynamic delimiters
- Structured prompt storage
- PromptBuilder behavior
- Compression model selection from active models
"""

import pytest

from arbitrium_core import Arbitrium
from arbitrium_core.domain.prompts import PromptBuilder, PromptFormatter
from arbitrium_core.infrastructure.config.defaults import (
    PROMPTS,
    select_model_with_highest_context,
)
from tests.integration.conftest import MockModel


class TestPromptStructure:
    """Test that prompts are properly structured."""

    def test_prompts_have_content_and_metadata(self) -> None:
        """All prompts should have 'content' and 'metadata' fields."""
        for prompt_type, prompt_data in PROMPTS.items():
            assert isinstance(
                prompt_data, dict
            ), f"Prompt '{prompt_type}' should be a dict"
            assert (
                "content" in prompt_data
            ), f"Prompt '{prompt_type}' missing 'content'"
            assert (
                "metadata" in prompt_data
            ), f"Prompt '{prompt_type}' missing 'metadata'"
            assert isinstance(
                prompt_data["content"], str
            ), f"Prompt '{prompt_type}' content should be string"
            assert isinstance(
                prompt_data["metadata"], dict
            ), f"Prompt '{prompt_type}' metadata should be dict"

    def test_prompts_metadata_has_version(self) -> None:
        """All prompt metadata should have version info."""
        for prompt_type, prompt_data in PROMPTS.items():
            metadata = prompt_data["metadata"]
            assert (
                "version" in metadata
            ), f"Prompt '{prompt_type}' metadata missing version"
            assert (
                "type" in metadata
            ), f"Prompt '{prompt_type}' metadata missing type"
            assert (
                "phase" in metadata
            ), f"Prompt '{prompt_type}' metadata missing phase"

    def test_prompts_content_not_empty(self) -> None:
        """All prompts should have non-empty content."""
        for prompt_type, prompt_data in PROMPTS.items():
            content = prompt_data["content"]
            assert (
                len(content) > 0
            ), f"Prompt '{prompt_type}' has empty content"
            assert (
                len(content) > 50
            ), f"Prompt '{prompt_type}' content seems too short"


class TestPromptFormatterBehavior:
    """Test that PromptFormatter works correctly."""

    def test_formatter_creates_delimiters(self) -> None:
        """Formatter should create proper section delimiters."""
        formatter = PromptFormatter("default")
        result = formatter.wrap_section("TEST", "content")

        # Should have BEGIN and END
        assert "BEGIN" in result
        assert "END" in result
        assert "TEST" in result
        assert "content" in result

    def test_formatter_different_styles_produce_different_output(
        self,
    ) -> None:
        """Different delimiter styles should produce different output."""
        default_formatter = PromptFormatter("default")
        compact_formatter = PromptFormatter("compact")

        default_result = default_formatter.wrap_section("TEST", "content")
        compact_result = compact_formatter.wrap_section("TEST", "content")

        # Results should be different (different delimiters)
        assert (
            default_result != compact_result
        ), "Different styles should produce different output"

        # But both should contain the content
        assert "content" in default_result
        assert "content" in compact_result

    def test_formatter_preserves_content(self) -> None:
        """Formatter should preserve the actual content unchanged."""
        formatter = PromptFormatter("default")
        test_content = (
            "This is my important content that must be preserved exactly."
        )

        result = formatter.wrap_section("SECTION", test_content)

        # Content should appear exactly as given
        assert test_content in result

    def test_formatter_handles_multiline_content(self) -> None:
        """Formatter should handle multiline content correctly."""
        formatter = PromptFormatter("default")
        multiline_content = """Line 1
Line 2
Line 3"""

        result = formatter.wrap_section("MULTILINE", multiline_content)

        # All lines should be present
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result


class TestPromptBuilderBehavior:
    """Test that PromptBuilder creates correct prompts."""

    def test_prompt_builder_creates_initial_prompt(self) -> None:
        """PromptBuilder should create a valid initial prompt."""
        formatter = PromptFormatter("default")
        builder = PromptBuilder(PROMPTS, formatter)

        result = builder.build_initial_prompt("What is the meaning of life?")

        # Should contain the question
        assert "What is the meaning of life?" in result

        # Should contain base instruction from PROMPTS
        assert "multiple perspectives" in result or "evidence-based" in result

        # Should have proper structure
        assert "BEGIN" in result
        assert "END" in result

    def test_prompt_builder_creates_feedback_prompt(self) -> None:
        """PromptBuilder should create a valid feedback prompt."""
        formatter = PromptFormatter("default")
        builder = PromptBuilder(PROMPTS, formatter)

        result = builder.build_feedback_prompt(
            initial_question="Test question",
            target_answer="Test answer",
            feedback_instruction="Provide feedback",
        )

        # Should contain all components
        assert "Test question" in result
        assert "Test answer" in result
        assert "Provide feedback" in result

    def test_prompt_builder_creates_improvement_prompt(self) -> None:
        """PromptBuilder should create a valid improvement prompt."""
        formatter = PromptFormatter("default")
        builder = PromptBuilder(PROMPTS, formatter)

        result = builder.build_improvement_prompt(
            initial_question="Question",
            own_answer="My answer",
            improvement_instruction="Improve it",
            kb_context="",
            improvement_context=None,
            other_responses=None,
            model=None,  # type: ignore
            display_name="TestModel",
        )

        # Should contain the components
        assert "Question" in result
        assert "My answer" in result
        assert "Improve it" in result


class TestCompressionModelSelection:
    """Test that compression model is selected from active models only."""

    def test_select_from_model_configs(self) -> None:
        """Should select highest context model from config dicts."""
        models_config = {
            "small": {"context_window": 4096},
            "medium": {"context_window": 8192},
            "large": {"context_window": 16384},
        }

        result = select_model_with_highest_context(models_config)

        assert result == "large", "Should select model with highest context"

    def test_select_from_model_instances(self) -> None:
        """Should select highest context model from BaseModel instances."""
        models = {
            "small": MockModel(model_name="small", context_window=4096),
            "large": MockModel(model_name="large", context_window=16384),
            "medium": MockModel(model_name="medium", context_window=8192),
        }

        result = select_model_with_highest_context(models)

        assert (
            result == "large"
        ), "Should select model instance with highest context"

    def test_select_returns_none_for_empty_dict(self) -> None:
        """Should return None when no models available."""
        result = select_model_with_highest_context({})

        assert result is None, "Should return None for empty models dict"

    def test_select_handles_models_without_context(self) -> None:
        """Should handle models that don't have context_window."""
        models = {
            "no_context": {},
            "with_context": {"context_window": 8192},
        }

        result = select_model_with_highest_context(models)

        assert (
            result == "with_context"
        ), "Should select model with context_window"

    @pytest.mark.asyncio
    async def test_compression_model_selected_from_active_models(
        self, basic_config: dict
    ) -> None:
        """Compression model should be selected from active models, not all defaults."""
        # Configure only specific models
        basic_config["models"] = {
            "phi3": {
                "provider": "mock",
                "model_name": "ollama/phi3",
                "display_name": "Phi-3",
                "temperature": 0.7,
                "max_tokens": 1024,
                "context_window": 4096,
            },
            "qwen": {
                "provider": "mock",
                "model_name": "ollama/qwen3:8b",
                "display_name": "Qwen 8B",
                "temperature": 0.7,
                "max_tokens": 2048,
                "context_window": 8192,  # Highest context
            },
        }
        basic_config["features"]["compression_model"] = None  # Auto-select

        # Create Arbitrium
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Check that compression model was set
        for _model_key, model in arbitrium.all_models.items():
            # Should be set to qwen's model_name (highest context in active models)
            if model.compression_model is not None:
                # The compression model should come from active models
                # NOT from some other default model like "deepseek-v3"
                assert model.compression_model in [
                    "ollama/phi3",
                    "ollama/qwen3:8b",
                ], (
                    f"Compression model should be from active models, "
                    f"got {model.compression_model}"
                )


class TestPromptSystemIntegration:
    """Test prompt system integration in actual tournament."""

    @pytest.mark.asyncio
    async def test_tournament_uses_structured_prompts(
        self, basic_config: dict
    ) -> None:
        """Tournament should use structured prompts from config."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "a": MockModel(model_name="a", display_name="A"),
            "b": MockModel(model_name="b", display_name="B"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Run tournament - should not crash and should use prompts
        result, metrics = await arbitrium.run_tournament("Test question")

        # Should complete successfully (validates prompts worked)
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_delimiters_appear_in_model_prompts(
        self, basic_config: dict
    ) -> None:
        """Generated prompts should contain formatted delimiters."""

        # Create a mock model that captures the prompt it receives
        class CapturingMockModel(MockModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.received_prompts = []

            async def generate(self, prompt: str):
                self.received_prompts.append(prompt)
                return await super().generate(prompt)

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        capturing_model = CapturingMockModel(
            model_name="capturing", display_name="Capturing Model"
        )

        mock_models = {
            "capturing": capturing_model,
            "other": MockModel(model_name="other", display_name="Other"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Run tournament
        await arbitrium.run_tournament("Test question")

        # Check that prompts were formatted with delimiters
        assert (
            len(capturing_model.received_prompts) > 0
        ), "Model should have received prompts"

        # At least one prompt should have delimiters
        has_delimiters = any(
            "BEGIN" in prompt and "END" in prompt
            for prompt in capturing_model.received_prompts
        )
        assert has_delimiters, "Prompts should contain BEGIN/END delimiters"


class TestDelimiterConsistency:
    """Test that delimiters are consistent and not hardcoded."""

    def test_no_hardcoded_delimiters_in_template_content(self) -> None:
        """Template content should not have hardcoded ========== delimiters."""
        from arbitrium_core.domain.prompts.templates import (
            EVALUATION_PROMPT_TEMPLATE,
            FEEDBACK_PROMPT_TEMPLATE,
            IMPROVEMENT_PROMPT_TEMPLATE,
            INITIAL_PROMPT_TEMPLATE,
        )

        templates = [
            INITIAL_PROMPT_TEMPLATE,
            FEEDBACK_PROMPT_TEMPLATE,
            IMPROVEMENT_PROMPT_TEMPLATE,
            EVALUATION_PROMPT_TEMPLATE,
        ]

        for template in templates:
            # Should not have hardcoded ========== BEGIN/END patterns
            assert (
                "==========" not in template
            ), f"Template should not have hardcoded delimiters: {template[:50]}..."

    def test_formatter_is_source_of_delimiters(self) -> None:
        """Delimiters should only come from PromptFormatter."""
        formatter = PromptFormatter("default")

        # Formatter should be responsible for creating delimiters
        result = formatter.wrap_section("TEST", "content")

        # Should have delimiter characters
        assert "=" in result, "Formatter should create delimiters"
        assert "BEGIN" in result
        assert "END" in result
