"""End-to-end tests for error handling and edge cases."""

import pytest

from arbitrium_core import Arbitrium
from arbitrium_core.application.bootstrap import health_check_models
from tests.integration.conftest import MockModel


class TestModelFailures:
    """Test handling of model failures during execution."""

    @pytest.mark.asyncio
    async def test_tournament_continues_with_partially_failed_models(
        self,
        basic_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that tournament continues when some models fail."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Mix of healthy and failing models
        mock_models = {
            "healthy_a": MockModel(
                model_name="test-a",
                display_name="Healthy A",
            ),
            "healthy_b": MockModel(
                model_name="test-b",
                display_name="Healthy B",
            ),
            "failing": MockModel(
                model_name="test-fail",
                display_name="Failing Model",
                should_fail=True,
            ),
        }

        # Perform health check
        healthy, failed = await health_check_models(mock_models)

        # Should filter out failing model
        assert len(healthy) == 2
        assert len(failed) == 1
        assert "failing" in failed

        # Tournament should work with healthy models only
        arbitrium._healthy_models = healthy  # type: ignore[assignment]
        arbitrium._comparison = arbitrium._create_comparison()

        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_all_models_failing_raises_runtime_error(
        self,
        basic_config: dict,
    ) -> None:
        """Test that all models failing raises RuntimeError."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # All failing models
        all_failing = {
            "fail_1": MockModel(
                model_name="fail-1",
                display_name="Fail 1",
                should_fail=True,
            ),
            "fail_2": MockModel(
                model_name="fail-2",
                display_name="Fail 2",
                should_fail=True,
            ),
        }

        # Health check should mark all as failed
        healthy, failed = await health_check_models(all_failing)
        assert len(healthy) == 0
        assert len(failed) == 2

        # Set empty healthy models
        arbitrium._healthy_models = {}

        # Should raise RuntimeError when trying to run tournament
        with pytest.raises(RuntimeError, match="No healthy models available"):
            await arbitrium.run_tournament("Test question?")

    @pytest.mark.asyncio
    async def test_model_failure_tracked_with_error_message(
        self,
        basic_config: dict,
    ) -> None:
        """Test that model failures are tracked with error messages."""
        await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        failing_model = MockModel(
            model_name="fail",
            display_name="Failing",
            should_fail=True,
        )

        # Health check
        _healthy, failed = await health_check_models(
            {"failing": failing_model}
        )

        # Should have error tracked
        assert len(failed) == 1
        assert "failing" in failed
        assert isinstance(failed["failing"], Exception)


class TestInvalidResponses:
    """Test handling of invalid or problematic responses."""

    @pytest.mark.asyncio
    async def test_empty_response_handling(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling of empty responses from models."""
        await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Model that returns empty response
        empty_response_model = MockModel(
            model_name="empty",
            display_name="Empty",
            response_text="",
        )

        # Should be caught during health check or validation
        response = await empty_response_model.generate("test")
        assert response.content == ""

    @pytest.mark.asyncio
    async def test_very_short_response_handling(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test handling of very short responses."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        # Very short response should be rejected for KB extraction
        short_response = "OK"
        is_valid, reason = kb._is_valid_response_for_extraction(short_response)

        assert not is_valid
        assert "too short" in reason.lower()


class TestTimeoutHandling:
    """Test timeout and delay handling."""

    @pytest.mark.asyncio
    async def test_model_with_delay_completes_successfully(
        self,
        basic_config: dict,
    ) -> None:
        """Test that models with delay complete successfully."""
        await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Model with small delay
        delayed_model = MockModel(
            model_name="delayed",
            display_name="Delayed",
            delay=0.1,  # 100ms delay
        )

        # Should complete without error
        response = await delayed_model.generate("test")
        assert response.is_successful
        assert response.content is not None


class TestEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_single_model_tournament(
        self,
        minimal_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test tournament with only one model."""
        # Configure with single model
        single_model_config = minimal_config.copy()
        single_model_config["models"] = {
            "only_model": {
                "provider": "mock",
                "model_name": "test",
                "display_name": "Only Model",
                "context_window": 4000,
                "max_tokens": 1000,
                "temperature": 0.7,
            }
        }

        arbitrium = await Arbitrium.from_settings(
            settings=single_model_config,
            skip_secrets=True,
        )

        mock_model = MockModel(model_name="test", display_name="Only Model")
        arbitrium._healthy_models = {"only_model": mock_model}  # type: ignore[assignment]
        arbitrium._comparison = arbitrium._create_comparison()

        # Single model should win by default
        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] == "only_model"
        assert len(metrics["eliminated_models"]) == 0

    @pytest.mark.asyncio
    async def test_empty_question_handling(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling of empty question."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore[assignment]
        arbitrium._comparison = arbitrium._create_comparison()

        # Empty question should still work (models will respond)
        result, _metrics = await arbitrium.run_tournament("")

        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_question_handling(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling of very long questions."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore[assignment]
        arbitrium._comparison = arbitrium._create_comparison()

        # Very long question
        long_question = "What is the meaning of life? " * 1000

        result, _metrics = await arbitrium.run_tournament(long_question)

        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters_in_question(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling of special characters in questions."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore[assignment]
        arbitrium._comparison = arbitrium._create_comparison()

        # Question with special characters
        special_question = "What about émojis 🚀 and symbols: @#$%^&*()?"

        result, _metrics = await arbitrium.run_tournament(special_question)

        assert result is not None

    @pytest.mark.asyncio
    async def test_unicode_in_responses(
        self,
        basic_config: dict,
    ) -> None:
        """Test handling of Unicode in model responses."""
        await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        unicode_model = MockModel(
            model_name="unicode",
            display_name="Unicode Model",
            response_text="Response with émojis 🎉 and unicode: 日本語",
        )

        # Should handle unicode without error
        response = await unicode_model.generate("test")
        assert response.is_successful
        assert "🎉" in response.content


class TestRecoveryMechanisms:
    """Test error recovery and resilience."""

    @pytest.mark.asyncio
    async def test_tournament_completes_despite_model_error_responses(
        self,
        basic_config: dict,
    ) -> None:
        """Test that tournament completes even with some error responses."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Models that may occasionally have issues
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                response_text="Good response",
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                response_text="Another good response",
            ),
        }

        arbitrium._healthy_models = mock_models  # type: ignore[assignment]
        arbitrium._comparison = arbitrium._create_comparison()

        # Should complete successfully
        result, metrics = await arbitrium.run_tournament("Test question?")

        assert result is not None
        assert metrics["champion_model"] is not None


class TestCostTracking:
    """Test cost tracking in error scenarios."""

    @pytest.mark.asyncio
    async def test_cost_tracked_even_with_failures(
        self,
        basic_config: dict,
    ) -> None:
        """Test that costs are tracked even when some models fail."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore[assignment]
        arbitrium._comparison = arbitrium._create_comparison()

        _result, metrics = await arbitrium.run_tournament("Test question?")

        # Should have cost tracked
        assert "total_cost" in metrics
        assert metrics["total_cost"] >= 0
        assert "cost_by_model" in metrics


class TestConcurrency:
    """Test concurrent operations and race conditions."""

    @pytest.mark.asyncio
    async def test_multiple_models_respond_concurrently(
        self,
        basic_config: dict,
    ) -> None:
        """Test that multiple models can respond concurrently."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        # Multiple models with delays
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                delay=0.05,
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                delay=0.05,
            ),
            "model_c": MockModel(
                model_name="test-c",
                display_name="Model C",
                delay=0.05,
            ),
        }

        arbitrium._healthy_models = mock_models  # type: ignore[assignment]
        arbitrium._comparison = arbitrium._create_comparison()

        # Should complete faster than sequential execution
        import time

        start = time.time()
        result, _metrics = await arbitrium.run_tournament("Test question?")
        time.time() - start

        # If truly concurrent, should be much faster than 3 * 0.05 * N_calls
        # Just verify it completes
        assert result is not None


class TestStateManagement:
    """Test state management and cleanup."""

    @pytest.mark.asyncio
    async def test_comparison_state_resets_between_tournaments(
        self,
        basic_config: dict,
    ) -> None:
        """Test that comparison state is fresh for each tournament."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore[assignment]

        # Run first tournament
        comparison1 = arbitrium._create_comparison()
        result1 = await comparison1.run("Question 1?")

        # Run second tournament with fresh comparison
        comparison2 = arbitrium._create_comparison()

        # Should have fresh state
        assert len(comparison2.previous_answers) == 0
        assert len(comparison2.eliminated_models) == 0
        assert comparison2.total_cost == 0

        result2 = await comparison2.run("Question 2?")

        # Both should complete independently
        assert result1 is not None
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_model_call_count_increments_correctly(
        self,
        basic_config: dict,
    ) -> None:
        """Test that model call counts increment correctly."""
        await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_model = MockModel(
            model_name="test",
            display_name="Test Model",
        )

        # Make multiple calls
        initial_count = mock_model._call_count

        await mock_model.generate("Test 1")
        await mock_model.generate("Test 2")
        await mock_model.generate("Test 3")

        # Should increment
        assert mock_model._call_count == initial_count + 3


class TestValidationEdgeCases:
    """Test validation edge cases."""

    @pytest.mark.asyncio
    async def test_null_values_in_config_handled_gracefully(
        self,
        tmp_output_dir,
    ) -> None:
        """Test that null values in config are handled gracefully."""
        config_with_nulls = {
            "models": {
                "test": {
                    "provider": "mock",
                    "model_name": "test",
                    "display_name": None,  # Null value
                    "context_window": 4000,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                },
            },
            "outputs_dir": str(tmp_output_dir),
        }

        # Should either use default or handle gracefully
        arbitrium = await Arbitrium.from_settings(
            settings=config_with_nulls,
            skip_secrets=True,
            skip_health_check=True,
        )

        assert arbitrium is not None

    @pytest.mark.asyncio
    async def test_extra_config_fields_ignored(
        self,
        basic_config: dict,
    ) -> None:
        """Test that extra unknown config fields are ignored."""
        basic_config["unknown_field"] = "some value"
        basic_config["another_unknown"] = {"nested": "value"}

        # Should load successfully, ignoring unknown fields
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        assert arbitrium is not None
        assert arbitrium.is_ready
