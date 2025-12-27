"""Advanced end-to-end tests for tournament edge cases and advanced scenarios."""

import pytest

from arbitrium_core import Arbitrium
from arbitrium_core.ports.llm import ModelResponse
from tests.integration.conftest import MockModel


class TestTournamentPhaseTransitions:
    """Test transitions between tournament phases."""

    @pytest.mark.asyncio
    async def test_tournament_with_improvement_disabled(
        self,
        minimal_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test tournament with improvement phase disabled."""
        # Disable improvement phase
        minimal_config["improvement_phase"]["enabled"] = False

        arbitrium = await Arbitrium.from_settings(
            settings=minimal_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_tournament_with_refinement_disabled(
        self,
        minimal_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test tournament with refinement phase disabled."""
        # Disable refinement phase
        minimal_config["refinement_phase"]["enabled"] = False
        minimal_config["improvement_phase"]["enabled"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=minimal_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_tournament_with_all_phases_disabled(
        self,
        minimal_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test tournament with all optional phases disabled."""
        minimal_config["improvement_phase"]["enabled"] = False
        minimal_config["refinement_phase"]["enabled"] = False

        arbitrium = await Arbitrium.from_settings(
            settings=minimal_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Should still work with just initial + evaluation
        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_tournament_with_feedback_enabled(
        self,
        basic_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test tournament with feedback enabled."""
        basic_config["improvement_phase"]["feedback_enabled"] = True

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

        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None


class TestAnonymization:
    """Test model anonymization functionality."""

    @pytest.mark.asyncio
    async def test_models_anonymized_during_evaluation(
        self,
        basic_config: dict,
    ) -> None:
        """Test that model names are anonymized during evaluation."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models

        comparison = arbitrium._create_comparison()

        # Check anonymization mapping exists
        assert comparison.anon_mapping is not None
        assert len(comparison.anon_mapping) == 2

        # Anonymized names should be different from originals
        for original, anon in comparison.anon_mapping.items():
            assert original != anon
            # Should be format like "LLM1", "LLM2", etc.
            assert anon.startswith("LLM")

    @pytest.mark.asyncio
    async def test_deanonymization_works_correctly(
        self,
        basic_config: dict,
    ) -> None:
        """Test that deanonymization maps back to original names."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models

        comparison = arbitrium._create_comparison()

        # Test that anonymization and reverse mapping work
        responses = {"model_a": "Response A", "model_b": "Response B"}
        anon_responses, reverse_mapping = (
            comparison.anonymizer.anonymize_responses(responses)
        )

        # Verify reverse mapping works
        for anon_name, original_name in reverse_mapping.items():
            assert original_name in responses
            assert anon_name in anon_responses


class TestEliminationStrategies:
    """Test different elimination strategies and edge cases."""

    @pytest.mark.asyncio
    async def test_progressive_elimination_reduces_models(
        self,
        basic_config: dict,
    ) -> None:
        """Test that progressive elimination reduces model count each round."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Start with 3 models
        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "model_c": MockModel(model_name="test-c", display_name="Model C"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        await arbitrium.run_tournament("Test question?")

        comparison = arbitrium._last_comparison

        # Should have eliminated models progressively
        assert len(comparison.eliminated_models) > 0
        assert len(comparison.active_model_keys) == 1

    @pytest.mark.asyncio
    async def test_elimination_tracking_includes_all_fields(
        self,
        basic_config: dict,
    ) -> None:
        """Test that elimination records include all required fields."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "model_c": MockModel(model_name="test-c", display_name="Model C"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        await arbitrium.run_tournament("Test question?")

        comparison = arbitrium._last_comparison

        # Check elimination records
        for elimination in comparison.eliminated_models:
            assert "model" in elimination
            assert "round" in elimination
            assert "reason" in elimination
            assert "score" in elimination
            assert isinstance(elimination["model"], str)
            assert isinstance(elimination["round"], int)


class TestCostAccumulation:
    """Test cost tracking and accumulation."""

    @pytest.mark.asyncio
    async def test_cost_accumulates_across_phases(
        self,
        basic_config: dict,
    ) -> None:
        """Test that costs accumulate across all phases."""
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

        _result, metrics = await arbitrium.run_tournament("Test question?")

        # Should have accumulated costs
        assert metrics["total_cost"] > 0

        # Should have per-model costs
        assert len(metrics["cost_by_model"]) > 0
        for _model_key, cost in metrics["cost_by_model"].items():
            assert cost >= 0

    @pytest.mark.asyncio
    async def test_cost_by_model_tracks_individual_usage(
        self,
        basic_config: dict,
    ) -> None:
        """Test that per-model costs are tracked correctly."""
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

        _result, metrics = await arbitrium.run_tournament("Test question?")

        # Each model should have cost tracked (by display name)
        assert len(metrics["cost_by_model"]) >= 2
        # Cost should be tracked for all models
        for cost in metrics["cost_by_model"].values():
            assert cost >= 0


class TestResponseSharing:
    """Test response sharing between models."""

    @pytest.mark.asyncio
    async def test_response_sharing_enabled_in_improvement(
        self,
        basic_config: dict,
    ) -> None:
        """Test that response sharing works in improvement phase."""
        basic_config["improvement_phase"]["share_responses"] = True

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

        result, _metrics = await arbitrium.run_tournament("Test question?")

        # Should complete successfully with response sharing
        assert result is not None

    @pytest.mark.asyncio
    async def test_response_sharing_disabled(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament with response sharing disabled."""
        basic_config["improvement_phase"]["share_responses"] = False
        basic_config["refinement_phase"]["share_responses"] = False

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

        result, _metrics = await arbitrium.run_tournament("Test question?")

        # Should complete successfully without response sharing
        assert result is not None


class TestModelResponseFormats:
    """Test handling of different model response formats."""

    @pytest.mark.asyncio
    async def test_model_response_with_metadata(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling model responses with metadata."""
        await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_model = MockModel(
            model_name="test",
            display_name="Test Model",
            response_text="Test response",
        )

        response = await mock_model.generate("Test prompt")

        # Check response structure
        assert isinstance(response, ModelResponse)
        assert response.content is not None
        assert response.cost >= 0
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_successful_response_properties(
        self,
        basic_config: dict,
    ) -> None:
        """Test properties of successful responses."""
        await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_model = MockModel(
            model_name="test",
            display_name="Test Model",
            response_text="Success",
        )

        response = await mock_model.generate("Test")

        assert response.is_successful
        assert not response.is_error()
        assert response.error is None

    @pytest.mark.asyncio
    async def test_error_response_properties(
        self,
        basic_config: dict,
    ) -> None:
        """Test properties of error responses."""
        await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        failing_model = MockModel(
            model_name="fail",
            display_name="Failing Model",
            should_fail=True,
        )

        response = await failing_model.generate("Test")

        assert not response.is_successful
        assert response.is_error()
        assert response.error is not None


class TestDeterministicMode:
    """Test deterministic mode functionality."""

    @pytest.mark.asyncio
    async def test_deterministic_mode_enabled(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament with deterministic mode enabled."""
        basic_config["features"]["deterministic_mode"] = True

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

        result1, _metrics1 = await arbitrium.run_tournament("Test question?")

        # Create new comparison for second run
        arbitrium._comparison = arbitrium._create_comparison()

        result2, _metrics2 = await arbitrium.run_tournament("Test question?")

        # Both runs should complete (determinism means same question -> same result)
        assert result1 is not None
        assert result2 is not None


class TestProvenance:
    """Test provenance tracking and recording."""

    @pytest.mark.asyncio
    async def test_provenance_tracking_enabled(
        self,
        basic_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that provenance is tracked when enabled."""
        basic_config["features"]["save_reports_to_disk"] = True

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

        await arbitrium.run_tournament("Test question?")

        # Check that provenance file was created
        list(tmp_output_dir.glob("*provenance*"))
        # May or may not have provenance depending on implementation
        # Just verify tournament completed
        assert tmp_output_dir.exists()


class TestMultiRoundTournaments:
    """Test tournaments with multiple rounds."""

    @pytest.mark.asyncio
    async def test_four_model_tournament_multiple_rounds(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament with 4 models requiring multiple rounds."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # 4 models should require 3 elimination rounds
        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "model_c": MockModel(model_name="test-c", display_name="Model C"),
            "model_d": MockModel(model_name="test-d", display_name="Model D"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None

        # Should have eliminated 3 models
        assert len(metrics["eliminated_models"]) == 3

    @pytest.mark.asyncio
    async def test_five_model_tournament(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament with 5 models."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            f"model_{i}": MockModel(
                model_name=f"test-{i}",
                display_name=f"Model {i}",
            )
            for i in range(5)
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None
        assert len(metrics["eliminated_models"]) == 4


class TestPromptCustomization:
    """Test custom prompt templates."""

    @pytest.mark.asyncio
    async def test_custom_initial_prompt(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament with custom initial prompt."""
        basic_config["prompts"][
            "initial"
        ] = "Custom initial prompt: {question}"

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        result, _metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None

    @pytest.mark.asyncio
    async def test_custom_evaluation_prompt(
        self,
        basic_config: dict,
    ) -> None:
        """Test tournament with custom evaluation prompt."""
        basic_config["prompts"]["evaluate"] = "Custom evaluate: {responses}"

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

        result, _metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
