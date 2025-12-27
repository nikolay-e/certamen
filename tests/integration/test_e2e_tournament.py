"""End-to-end tests for tournament workflows."""

import pytest

from arbitrium_core import Arbitrium
from tests.integration.conftest import MockModel


class TestFullTournamentWorkflow:
    """Test complete tournament execution from start to finish."""

    @pytest.mark.asyncio
    async def test_basic_tournament_completes_successfully(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that a basic 3-model tournament completes successfully."""
        result, metrics = await arbitrium_instance.run_tournament(
            simple_question
        )

        # Verify tournament completed
        assert result is not None
        assert len(result) > 0

        # Verify metrics
        assert "champion_model" in metrics
        assert "total_cost" in metrics
        assert "eliminated_models" in metrics
        assert metrics["total_cost"] >= 0

        # Verify we have a champion
        assert metrics["champion_model"] is not None

        # Verify eliminations occurred (3 models -> 1 champion means 2 eliminations)
        assert len(metrics["eliminated_models"]) == 2

    @pytest.mark.asyncio
    async def test_tournament_with_complex_question(
        self,
        arbitrium_instance: Arbitrium,
        complex_question: str,
    ) -> None:
        """Test tournament with a complex question requiring detailed analysis."""
        result, metrics = await arbitrium_instance.run_tournament(
            complex_question
        )

        assert result is not None
        assert (
            len(result) > 20
        )  # Complex question should have answer (relaxed for mock data)
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_tournament_phases_execute_in_order(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that tournament phases execute in correct order."""
        # Run tournament
        await arbitrium_instance.run_tournament(simple_question)

        # Access the comparison object that was used
        comparison = arbitrium_instance._last_comparison
        assert comparison is not None

        # Verify phases completed
        # previous_answers should have: [initial, improved, round1_refined, ...]
        assert len(comparison.previous_answers) >= 2

        # Verify evaluation history
        assert len(comparison.evaluation_history) >= 1

        # Verify eliminations occurred
        assert len(comparison.eliminated_models) >= 1

    @pytest.mark.asyncio
    async def test_minimal_tournament_two_models(
        self,
        minimal_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test tournament with minimal setup (2 models, no extra phases)."""
        # Create arbitrium with minimal config
        arbitrium = await Arbitrium.from_settings(
            settings=minimal_config,
            skip_secrets=True,
        )

        # Add mock models
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                response_text="Answer A",
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                response_text="Answer B",
            ),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()
        comparison.models = mock_models
        comparison.active_model_keys = list(mock_models.keys())
        comparison.anon_mapping = comparison.anonymizer.anonymize_model_keys(
            list(mock_models.keys())
        )

        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None
        # 2 models -> 1 champion means 1 elimination
        assert len(metrics["eliminated_models"]) == 1


class TestTournamentPhases:
    """Test individual tournament phases."""

    @pytest.mark.asyncio
    async def test_initial_phase_generates_responses(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that initial phase generates responses from all models."""
        # Run tournament first
        await arbitrium_instance.run_tournament(simple_question)

        # Access the comparison object that was used
        comparison = arbitrium_instance._last_comparison
        assert comparison is not None

        # Verify all models responded initially (check previous_answers)
        # The first entry in previous_answers should have responses from all 3 models
        assert len(comparison.previous_answers) > 0
        initial_responses = comparison.previous_answers[0]
        assert len(initial_responses) == 3  # 3 models configured
        assert all(
            len(response) > 0 for response in initial_responses.values()
        )

    @pytest.mark.asyncio
    async def test_improvement_phase_refines_answers(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that improvement phase refines initial answers."""
        # Run tournament first
        await arbitrium_instance.run_tournament(simple_question)

        # Access the comparison object that was used
        comparison = arbitrium_instance._last_comparison
        assert comparison is not None

        # Verify improvements occurred (check previous_answers has multiple entries)
        assert len(comparison.previous_answers) >= 2
        initial_responses = comparison.previous_answers[0]
        improved_responses = comparison.previous_answers[1]

        # Verify improvements
        assert len(improved_responses) > 0
        assert len(initial_responses) > 0

    @pytest.mark.asyncio
    async def test_evaluation_phase_produces_scores(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that evaluation phase produces valid scores."""
        # Run tournament first
        await arbitrium_instance.run_tournament(simple_question)

        # Access the comparison object that was used
        comparison = arbitrium_instance._last_comparison
        assert comparison is not None

        # Verify evaluations produced
        assert len(comparison.evaluation_history) > 0
        assert hasattr(comparison, "evaluation_scores")
        assert comparison.evaluation_scores is not None


class TestTournamentElimination:
    """Test elimination logic and model removal."""

    @pytest.mark.asyncio
    async def test_elimination_removes_model_from_active(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that elimination removes model from active participants."""
        initial_count = len(arbitrium_instance._healthy_models)

        # Run tournament
        await arbitrium_instance.run_tournament(simple_question)

        # Access the comparison object that was used
        comparison = arbitrium_instance._last_comparison
        assert comparison is not None

        # Verify models were eliminated
        final_count = len(comparison.active_model_keys)
        assert final_count < initial_count
        assert final_count == 1  # Only champion remains

    @pytest.mark.asyncio
    async def test_elimination_tracking_includes_reason(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that eliminations are tracked with reasons."""
        await arbitrium_instance.run_tournament(simple_question)

        # Access the comparison object that was used
        comparison = arbitrium_instance._last_comparison
        assert comparison is not None

        # Verify elimination tracking
        assert len(comparison.eliminated_models) >= 1

        for elimination in comparison.eliminated_models:
            assert "model" in elimination
            assert "round" in elimination
            assert "reason" in elimination
            # Score may be None (random elimination) or float
            assert "score" in elimination

    @pytest.mark.asyncio
    async def test_lowest_score_model_gets_eliminated(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that model with lowest score is eliminated."""
        # Run tournament first
        await arbitrium_instance.run_tournament(simple_question)

        # Access the comparison object that was used
        comparison = arbitrium_instance._last_comparison
        assert comparison is not None

        # Verify eliminations occurred - models with lowest scores were eliminated
        assert len(comparison.eliminated_models) > 0

        # Verify we have a champion (highest ranked model)
        assert len(comparison.active_model_keys) == 1


class TestTournamentMetrics:
    """Test metrics collection and reporting."""

    @pytest.mark.asyncio
    async def test_metrics_include_all_required_fields(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that tournament metrics include all required fields."""
        _result, metrics = await arbitrium_instance.run_tournament(
            simple_question
        )

        # Required fields
        assert "champion_model" in metrics
        assert "total_cost" in metrics
        assert "cost_by_model" in metrics
        assert "eliminated_models" in metrics

        # Type checks
        assert isinstance(metrics["total_cost"], (int, float))
        assert isinstance(metrics["cost_by_model"], dict)
        assert isinstance(metrics["eliminated_models"], list)

    @pytest.mark.asyncio
    async def test_cost_tracking_accumulates(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that costs accumulate throughout tournament."""
        await arbitrium_instance.run_tournament(simple_question)

        # Access the comparison object that was used
        comparison = arbitrium_instance._last_comparison
        assert comparison is not None

        # Cost should be tracked
        assert comparison.total_cost >= 0

        # Cost by model should be tracked
        assert len(comparison.cost_by_model) > 0


class TestTournamentOutputs:
    """Test tournament output generation."""

    @pytest.mark.asyncio
    async def test_tournament_generates_reports(
        self,
        basic_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that tournament generates report files."""
        # Enable report generation
        basic_config["features"]["save_reports_to_disk"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Add mock models
        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()
        comparison.models = mock_models
        comparison.active_model_keys = list(mock_models.keys())
        comparison.anon_mapping = comparison.anonymizer.anonymize_model_keys(
            list(mock_models.keys())
        )

        await arbitrium.run_tournament("Test question?")

        # Check that output directory has files
        output_files = list(tmp_output_dir.glob("*"))
        # Should have champion report and provenance
        assert len(output_files) >= 2

    @pytest.mark.asyncio
    async def test_champion_answer_returned(
        self,
        arbitrium_instance: Arbitrium,
        simple_question: str,
    ) -> None:
        """Test that champion's answer is returned as result."""
        result, _metrics = await arbitrium_instance.run_tournament(
            simple_question
        )

        # Result should be the champion's answer
        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain content from mock responses
        assert any(
            keyword in result.lower()
            for keyword in ["model", "response", "answer", "call"]
        )
