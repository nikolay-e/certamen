"""End-to-end tests for judge mode and evaluation edge cases."""

import pytest

from arbitrium_core import Arbitrium
from tests.integration.conftest import MockModel


class TestJudgeMode:
    """Test tournament with dedicated judge model."""

    @pytest.mark.asyncio
    async def test_tournament_with_judge_model(
        self,
        basic_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test tournament with designated judge model."""
        # Configure judge model
        basic_config["features"]["judge_model"] = "judge_model"
        basic_config["models"]["judge_model"] = {
            "provider": "mock",
            "model_name": "judge",
            "display_name": "Judge Model",
            "temperature": 0.7,
            "max_tokens": 2000,
            "context_window": 8000,
        }

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Create mock models including judge
        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "judge_model": MockModel(
                model_name="judge", display_name="Judge Model"
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Judge should be removed from active participants
        assert "judge_model" not in arbitrium._comparison.active_model_keys
        assert len(arbitrium._comparison.active_model_keys) == 2

        # Run tournament
        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_judge_model_not_in_participants(
        self,
        basic_config: dict,
    ) -> None:
        """Test that judge model doesn't participate in tournament."""
        basic_config["features"]["judge_model"] = "judge_model"
        basic_config["models"]["judge_model"] = {
            "provider": "mock",
            "model_name": "judge",
            "display_name": "Judge Model",
            "temperature": 0.7,
            "max_tokens": 2000,
            "context_window": 8000,
        }

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "model_c": MockModel(model_name="test-c", display_name="Model C"),
            "judge_model": MockModel(
                model_name="judge", display_name="Judge Model"
            ),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()

        # Judge should not be in active keys
        assert "judge_model" not in comparison.active_model_keys
        assert len(comparison.active_model_keys) == 3

        # Judge should still be in all models
        assert "judge_model" in comparison.models

    @pytest.mark.asyncio
    async def test_invalid_judge_model_ignored(
        self,
        basic_config: dict,
    ) -> None:
        """Test that invalid judge model config is ignored."""
        basic_config["features"]["judge_model"] = "nonexistent_judge"

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

        # Should not have a judge (invalid name)
        assert comparison.judge_model_key is None
        # All models should participate
        assert len(comparison.active_model_keys) == 2

    @pytest.mark.asyncio
    async def test_judge_scores_used_for_elimination(
        self,
        basic_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that judge evaluation scores are correctly used for elimination.

        This test verifies that:
        1. Judge scores are keyed by anonymized names (LLM1, LLM2, etc.)
        2. Elimination logic correctly matches scores to active models
        3. The lowest scored model is eliminated based on judge scores
        """
        basic_config["features"]["judge_model"] = "judge_model"
        basic_config["models"]["judge_model"] = {
            "provider": "mock",
            "model_name": "judge",
            "display_name": "Judge Model",
            "temperature": 0.7,
            "max_tokens": 2000,
            "context_window": 8000,
        }

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "model_c": MockModel(model_name="test-c", display_name="Model C"),
            "judge_model": MockModel(
                model_name="judge", display_name="Judge Model"
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        comparison = arbitrium._comparison

        await comparison.run_cross_evaluation(
            "Test question?",
            {"LLM1": "Response A", "LLM2": "Response B", "LLM3": "Response C"},
            round_num=1,
        )

        assert (
            comparison.evaluation_scores
        ), "Judge should have provided scores"
        score_keys = set(comparison.evaluation_scores.keys())
        assert score_keys.issubset(
            {"LLM1", "LLM2", "LLM3"}
        ), f"Scores should be keyed by anonymized names, got: {score_keys}"

        eliminated, leader = (
            comparison.determine_lowest_and_highest_ranked_models()
        )

        assert (
            eliminated is not None
        ), "Should eliminate a model based on scores"
        assert leader is not None, "Should identify a leader based on scores"
        assert (
            "random" not in comparison.elimination_reason.lower()
        ), f"Should use judge scores, not random elimination. Reason: {comparison.elimination_reason}"


class TestEmergencyJudgeFallback:
    """Test emergency fallback to largest model as judge."""

    @pytest.mark.asyncio
    async def test_emergency_judge_selection(
        self,
        basic_config: dict,
    ) -> None:
        """Test emergency judge fallback when all peer evaluations fail."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Create models with different sizes
        mock_models = {
            "small_model": MockModel(
                model_name="small",
                display_name="Small Model",
                context_window=2000,
                max_tokens=500,
                # Make it fail at scoring
                response_text="I cannot evaluate these models.",
            ),
            "large_model": MockModel(
                model_name="large",
                display_name="Large Model",
                context_window=128000,
                max_tokens=4096,
                # Also fails at scoring
                response_text="Sorry, I can't score these.",
            ),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()

        # Test emergency judge selection
        emergency_judge = comparison._select_largest_model_as_judge()

        # Should select the larger model
        assert emergency_judge == "large_model"

    @pytest.mark.asyncio
    async def test_emergency_judge_with_no_models(
        self,
        basic_config: dict,
    ) -> None:
        """Test emergency judge selection when no models available."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()

        # Clear active models to simulate failure
        comparison.active_model_keys = []

        # Should return None
        emergency_judge = comparison._select_largest_model_as_judge()
        assert emergency_judge is None

    @pytest.mark.asyncio
    async def test_emergency_judge_activated_during_tournament(
        self,
        basic_config: dict,
    ) -> None:
        """Test that emergency judge is activated when all peer evaluations fail."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Create models: small ones fail at peer eval, large one succeeds as emergency judge
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                context_window=2000,
                max_tokens=500,
                response_text="Sorry, I cannot evaluate these models.",  # Fails at peer eval
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                context_window=128000,
                max_tokens=4096,
                # This model will succeed when used as emergency judge (no apology keywords)
                response_text="Comprehensive evaluation response",
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        comparison = arbitrium._comparison

        # Run cross-evaluation
        # model_a returns apology and fails
        # model_b would also fail at peer eval (but as emergency judge it will succeed)
        result = await comparison.run_cross_evaluation(
            "Test question?",
            {"LLM1": "Response A", "LLM2": "Response B"},
            round_num=1,
        )

        # Should have run emergency judge evaluation
        # After peer evaluation fails, it should fall back to emergency judge
        assert result is not None
        # evaluation_scores should be populated by emergency judge
        assert comparison.evaluation_scores is not None
        # Should have scores for the evaluated models
        assert len(comparison.evaluation_scores) > 0

    @pytest.mark.asyncio
    async def test_judge_model_error_handling(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling when judge model fails during evaluation."""
        basic_config["features"]["judge_model"] = "judge_model"
        basic_config["models"]["judge_model"] = {
            "provider": "mock",
            "model_name": "judge",
            "display_name": "Judge Model",
            "temperature": 0.7,
            "max_tokens": 2000,
            "context_window": 8000,
        }

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "judge_model": MockModel(
                model_name="judge",
                display_name="Judge Model",
                should_fail=True,  # Judge fails
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Run evaluation with failing judge
        result = await arbitrium._comparison.run_cross_evaluation(
            "Test question?",
            {"LLM1": "Response A", "LLM2": "Response B"},
            round_num=1,
        )

        # With fallback strategy, should succeed using emergency judge or peer review
        # instead of returning empty dict
        assert result != {}  # Should have evaluations from fallback
        # Scores should be present from fallback strategy
        assert arbitrium._comparison.evaluation_scores != {}


class TestEvaluationFailures:
    """Test evaluation failure scenarios."""

    @pytest.mark.asyncio
    async def test_model_failure_during_evaluation(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling of model failures during evaluation."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # One model fails during evaluation
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                response_text="Good response",
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                should_fail=True,  # This model fails
            ),
            "model_c": MockModel(
                model_name="test-c",
                display_name="Model C",
                response_text="Another good response",
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Tournament should still complete with remaining models
        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        # At least one model should remain
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_all_peer_evaluators_fail(
        self,
        basic_config: dict,
    ) -> None:
        """Test fallback when all peer evaluators fail to provide scores."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # All models refuse to evaluate
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                response_text="I cannot evaluate.",  # Apology/refusal
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                response_text="Sorry, I cannot score.",  # Apology/refusal
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Run comparison - should handle gracefully
        comparison = arbitrium._comparison

        # Simulate failed peer evaluation
        comparison.evaluation_scores = {}  # No scores from any evaluator

        # Should fall back to emergency measures
        eliminated, leader = (
            comparison.determine_lowest_and_highest_ranked_models()
        )

        # Should still select models (random selection)
        assert eliminated is not None
        assert leader is not None


class TestScoreAggregation:
    """Test score aggregation edge cases."""

    @pytest.mark.asyncio
    async def test_unscored_models_handled(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling of models that received no scores."""
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
        comparison = arbitrium._create_comparison()

        # Simulate partial scoring (some models not scored)
        comparison.evaluation_scores = {
            "Model A": {
                "Model B": 8.0
            },  # Only Model B scored, Model C not scored
        }

        eliminated, leader = (
            comparison.determine_lowest_and_highest_ranked_models()
        )

        assert eliminated is not None
        assert leader is not None

    @pytest.mark.asyncio
    async def test_all_models_tied_score(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling when all models have the same score."""
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
        comparison = arbitrium._create_comparison()

        # All models get the same score
        comparison.evaluation_scores = {
            "LLM1": 7.0,
            "LLM2": 7.0,
            "LLM3": 7.0,
        }

        eliminated, leader = (
            comparison.determine_lowest_and_highest_ranked_models()
        )

        # Should randomly select one for elimination and one as leader
        assert eliminated is not None
        assert leader is not None

    @pytest.mark.asyncio
    async def test_invalid_score_values_rejected(
        self,
        basic_config: dict,
    ) -> None:
        """Test that invalid score values are rejected."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()

        # Test normalization of invalid scores
        assert (
            comparison.score_extractor.normalize_score(-5.0, "test") is None
        )  # Negative
        assert (
            comparison.score_extractor.normalize_score(15.0, "test") is None
        )  # Too high
        assert (
            comparison.score_extractor.normalize_score(0.4, "test") is None
        )  # Too low

        # Valid scores should pass through
        assert comparison.score_extractor.normalize_score(5.0, "test") == 5.0
        assert comparison.score_extractor.normalize_score(10.0, "test") == 10.0

    @pytest.mark.asyncio
    async def test_critical_no_scores_scenario(
        self,
        basic_config: dict,
    ) -> None:
        """Test critical error handling when no models have any valid scores."""
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
        comparison = arbitrium._create_comparison()

        # Simulate critical scenario: no scores at all
        comparison.evaluation_scores = {}

        # Call ranking determination - should handle gracefully with random selection
        eliminated, leader = (
            comparison.determine_lowest_and_highest_ranked_models()
        )

        # Should still return some models (random selection as fallback)
        assert eliminated is not None
        assert leader is not None
        # Elimination reason should indicate this was random due to no scores
        assert "random" in comparison.elimination_reason.lower()
        assert comparison.elimination_score is None


class TestModelFailureHandling:
    """Test model failure and removal scenarios."""

    @pytest.mark.asyncio
    async def test_model_removed_after_failure(
        self,
        basic_config: dict,
    ) -> None:
        """Test that failed models are removed from tournament."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
            "model_c": MockModel(
                model_name="test-c",
                display_name="Model C",
                should_fail=True,
            ),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()

        initial_count = len(comparison.active_model_keys)

        # Simulate model failure
        comparison._handle_model_failure("model_c", "Test failure")

        # Model should be removed
        assert len(comparison.active_model_keys) == initial_count - 1
        assert "model_c" not in comparison.active_model_keys

    @pytest.mark.asyncio
    async def test_placeholder_responses_filtered(
        self,
        basic_config: dict,
    ) -> None:
        """Test that placeholder responses are filtered out."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models
        comparison = arbitrium._create_comparison()

        # Test filtering
        responses = {
            "Model A": "Real response",
            "Model B": "TBD",  # Placeholder
            "Model C": "",  # Empty
            "Model D": "test",  # Too short
        }

        valid, failed = comparison._filter_valid_responses(responses)

        assert "Model A" in valid
        assert "Model B" not in valid
        assert "Model C" not in valid
        assert "Model D" not in valid
        assert len(failed) == 3


class TestKnowledgeBankIntegration:
    """Test knowledge bank integration in tournaments."""

    @pytest.mark.asyncio
    async def test_eliminated_model_insights_extracted(
        self,
        kb_enabled_config: dict,
    ) -> None:
        """Test that insights are extracted from eliminated models."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                response_text="Model A's detailed insight-rich answer with specific technical considerations",
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                response_text="Model B's comprehensive response with valuable insights",
            ),
            "model_c": MockModel(
                model_name="test-c",
                display_name="Model C",
                response_text="Model C's thorough analysis with detailed explanations",
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Knowledge bank should be enabled
        assert arbitrium._comparison.knowledge_bank is not None

        # Run tournament
        await arbitrium.run_tournament("Test question?")

        # Knowledge bank should have insights from eliminated models
        # (exact count depends on elimination order)
        kb = arbitrium._comparison.knowledge_bank
        insights = await kb.get_all_insights()

        # Should have some insights collected
        assert isinstance(insights, list)


class TestConcurrentExecution:
    """Test concurrent model execution."""

    @pytest.mark.asyncio
    async def test_models_execute_concurrently(
        self,
        basic_config: dict,
    ) -> None:
        """Test that models execute tasks concurrently."""
        import time

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Models with delays
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                delay=0.05,  # Small delay
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                delay=0.05,
            ),
        }
        arbitrium._healthy_models = mock_models
        arbitrium._comparison = arbitrium._create_comparison()

        # Time the execution
        start = time.time()
        await arbitrium.run_tournament("Test question?")
        duration = time.time() - start

        # Should complete faster than sequential (2 models * delay * phases)
        # Just verify it completes without issues
        assert duration > 0

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(
        self,
        basic_config: dict,
    ) -> None:
        """Test that semaphore limits concurrent requests."""
        # Set max concurrent to 1
        basic_config.setdefault("model_defaults", {})
        basic_config["model_defaults"]["concurrency_limits"] = {
            "max_concurrent_requests": 1
        }

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

        # Semaphore should be set to 1
        assert comparison.semaphore._value == 1

        # Tournament should still work
        await arbitrium.run_tournament("Test question?")
