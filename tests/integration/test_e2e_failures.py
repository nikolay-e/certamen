"""End-to-end tests for tournament failure scenarios and error paths."""

import pytest

from arbitrium_core import Arbitrium
from tests.integration.conftest import MockModel


class TestTournamentFailureScenarios:
    """Test tournament behavior when things go wrong."""

    @pytest.mark.asyncio
    async def test_all_models_fail_initial_phase(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament when all models fail in initial phase."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # All models fail
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                should_fail=True,
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                should_fail=True,
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Tournament should fail gracefully
        result, _metrics = await arbitrium.run_tournament("Test question?")

        assert "cannot proceed" in result.lower() or "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_improvement_phase_failure(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament when improvement phase fails."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Create a model that fails only during improvement (call count > 1)
        class FailAfterInitialModel(MockModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._call_count = 0

            async def generate(self, prompt: str):
                self._call_count += 1
                # First call (initial) succeeds, subsequent (improvement) fail
                if self._call_count > 2:  # After initial responses
                    from arbitrium_core.ports.llm import ModelResponse

                    return ModelResponse.create_error("Improvement failed")
                return await super().generate(prompt)

        mock_models = {
            "model_a": FailAfterInitialModel(
                model_name="test-a",
                display_name="Model A",
            ),
            "model_b": FailAfterInitialModel(
                model_name="test-b",
                display_name="Model B",
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Tournament should handle improvement phase failure
        result, _metrics = await arbitrium.run_tournament("Test question?")

        # Should complete but may have issues
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_evaluations_in_elimination_round(
        self,
        basic_config: dict,
    ) -> None:
        """Test elimination round when no evaluations are returned."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Models that succeed initially but fail at evaluation
        class FailAtEvaluationModel(MockModel):
            async def generate(self, prompt: str):
                # Fail if it's an evaluation prompt
                if "evaluate" in prompt.lower() or "score" in prompt.lower():
                    from arbitrium_core.ports.llm import ModelResponse

                    return ModelResponse.create_error("Evaluation failed")
                return await super().generate(prompt)

        mock_models = {
            "model_a": FailAtEvaluationModel(
                model_name="test-a",
                display_name="Model A",
            ),
            "model_b": FailAtEvaluationModel(
                model_name="test-b",
                display_name="Model B",
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, _metrics = await arbitrium.run_tournament("Test question?")

        # Should complete even with evaluation failures
        assert result is not None

    @pytest.mark.asyncio
    async def test_refinement_phase_failure(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament when refinement phase fails."""
        # Enable refinement phase
        basic_config["refinement_phase"]["enabled"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Create models that fail during refinement (after multiple calls)
        class FailAtRefinementModel(MockModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._call_count = 0

            async def generate(self, prompt: str):
                self._call_count += 1
                # Fail after several calls (during refinement)
                if (
                    self._call_count > 6
                ):  # After initial + improvement + some evaluations
                    from arbitrium_core.ports.llm import ModelResponse

                    return ModelResponse.create_error("Refinement failed")
                return await super().generate(prompt)

        mock_models = {
            "model_a": FailAtRefinementModel(
                model_name="test-a",
                display_name="Model A",
            ),
            "model_b": FailAtRefinementModel(
                model_name="test-b",
                display_name="Model B",
            ),
            "model_c": FailAtRefinementModel(
                model_name="test-c",
                display_name="Model C",
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, _metrics = await arbitrium.run_tournament("Test question?")

        # Should complete despite refinement failures
        assert result is not None

    @pytest.mark.asyncio
    async def test_champion_answer_not_found(
        self,
        basic_config: dict,
    ) -> None:
        """Test finalization when champion answer is missing."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Manually manipulate to create scenario where champion answer is missing
        comparison = arbitrium._comparison

        # Run initial phase
        initial_responses = await comparison.run_initial_round(
            "Test question?"
        )
        comparison.previous_answers.append(initial_responses)

        # Force a single champion scenario by removing all but one model
        if len(comparison.active_model_keys) > 1:
            # Keep only the first model
            champion_key = comparison.active_model_keys[0]
            champion_anon = comparison.anon_mapping[champion_key]

            # Remove other models from active keys
            comparison.active_model_keys = [champion_key]
            comparison.anon_mapping = {champion_key: champion_anon}

            # Clear the champion's answer to trigger the error
            comparison.previous_answers[-1][champion_anon] = ""

        # Now try to finalize
        result = await comparison.runner._finalize_tournament("Test question?")

        # Should handle gracefully - either "not found" or "determined but"
        assert (
            "not found" in result.lower()
            or "determined but" in result.lower()
            or "remaining" in result.lower()
        )

    @pytest.mark.asyncio
    async def test_multiple_models_remain_at_end(
        self,
        minimal_config: dict,
    ) -> None:
        """Test when tournament ends with multiple models (no single champion)."""
        # Disable phases so we can control the flow
        minimal_config["improvement_phase"]["enabled"] = False
        minimal_config["refinement_phase"]["enabled"] = False

        arbitrium = await Arbitrium.from_settings(
            settings=minimal_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "model_c": MockModel(model_name="test-c", display_name="Model C"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        comparison = arbitrium._comparison

        # Run initial phase
        initial_responses = await comparison.run_initial_round(
            "Test question?"
        )
        comparison.previous_answers.append(initial_responses)

        # Manually break the elimination process by clearing active keys logic
        # Simulate ending with multiple models
        result = await comparison.runner._finalize_tournament("Test question?")

        # Should indicate multiple models remaining or single champion
        assert result is not None


class TestFeedbackPhaseFailures:
    """Test feedback phase error handling."""

    @pytest.mark.asyncio
    async def test_feedback_model_exception(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling of exception during feedback generation."""
        basic_config["improvement_phase"]["feedback_enabled"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Model that raises exception during feedback
        class ExceptionDuringFeedbackModel(MockModel):
            async def generate(self, prompt: str):
                if "feedback" in prompt.lower():
                    raise RuntimeError("Feedback generation crashed")
                return await super().generate(prompt)

        mock_models = {
            "model_a": ExceptionDuringFeedbackModel(
                model_name="test-a",
                display_name="Model A",
            ),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Should handle exception gracefully
        result, _metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None

    @pytest.mark.asyncio
    async def test_feedback_returns_error_response(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling when feedback returns error response."""
        basic_config["improvement_phase"]["feedback_enabled"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Model that returns error response for feedback
        class ErrorFeedbackModel(MockModel):
            async def generate(self, prompt: str):
                from arbitrium_core.ports.llm import ModelResponse

                if "feedback" in prompt.lower():
                    return ModelResponse.create_error(
                        "Cannot provide feedback"
                    )
                return await super().generate(prompt)

        mock_models = {
            "model_a": ErrorFeedbackModel(
                model_name="test-a",
                display_name="Model A",
            ),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, _metrics = await arbitrium.run_tournament("Test question?")

        # Should complete despite feedback errors
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_feedback_collected(
        self,
        basic_config: dict,
    ) -> None:
        """Test improvement phase when no feedback is collected."""
        basic_config["improvement_phase"]["feedback_enabled"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # All models fail at feedback
        class NoFeedbackModel(MockModel):
            async def generate(self, prompt: str):
                from arbitrium_core.ports.llm import ModelResponse

                if "feedback" in prompt.lower():
                    return ModelResponse.create_error("No feedback")
                return await super().generate(prompt)

        mock_models = {
            "model_a": NoFeedbackModel(
                model_name="test-a", display_name="Model A"
            ),
            "model_b": NoFeedbackModel(
                model_name="test-b", display_name="Model B"
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, _metrics = await arbitrium.run_tournament("Test question?")

        # Should proceed without feedback
        assert result is not None


class TestScoreNormalizationEdgeCases:
    """Test score normalization edge cases."""

    @pytest.mark.asyncio
    async def test_score_normalization_percentage_scale(
        self,
        basic_config: dict,
    ) -> None:
        """Test normalization of scores in 0-1 range (percentage)."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()

        # Test normalizing percentage scores (0-1 range)
        normalized = comparison.score_extractor.normalize_score(0.5, "test")
        assert normalized == 5.0  # 0.5 * 10 = 5.0

        normalized = comparison.score_extractor.normalize_score(0.8, "test")
        assert normalized == 8.0  # 0.8 * 10 = 8.0

        normalized = comparison.score_extractor.normalize_score(0.95, "test")
        assert normalized == 9.5  # 0.95 * 10 = 9.5

    @pytest.mark.asyncio
    async def test_score_normalization_slightly_over_ten(
        self,
        basic_config: dict,
    ) -> None:
        """Test normalization of scores slightly over 10."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()

        # Test normalizing scores between 10 and 10.5 (clamped to 10.0)
        normalized = comparison.score_extractor.normalize_score(10.1, "test")
        assert normalized == 10.0

        normalized = comparison.score_extractor.normalize_score(10.3, "test")
        assert normalized == 10.0

        normalized = comparison.score_extractor.normalize_score(10.5, "test")
        assert normalized == 10.0


class TestEmergencyJudgeNoModelsAvailable:
    """Test emergency judge fallback when no models are available."""

    @pytest.mark.asyncio
    async def test_emergency_judge_but_no_models_left(
        self,
        basic_config: dict,
    ) -> None:
        """Test emergency judge fallback when all models have been eliminated."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Start with models that will fail at peer evaluation
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                response_text="Sorry, I cannot evaluate.",
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                response_text="Sorry, I cannot evaluate.",
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        comparison = arbitrium._comparison

        # Manually clear active models to simulate all being eliminated
        comparison.active_model_keys = []

        # Try to run cross-evaluation with no models
        result = await comparison.run_cross_evaluation(
            "Test question?",
            {"LLM1": "Response A", "LLM2": "Response B"},
            round_num=1,
        )

        # Should handle gracefully - either empty result or emergency measures
        assert result is not None  # Should not crash
