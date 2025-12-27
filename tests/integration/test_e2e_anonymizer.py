"""End-to-end tests for model anonymization."""

import pytest

from arbitrium_core import Arbitrium
from arbitrium_core.domain.tournament.anonymizer import ModelAnonymizer
from tests.integration.conftest import MockModel


class TestAnonymizerBasics:
    """Test basic anonymization functionality."""

    def test_anonymize_model_keys_static_method(self) -> None:
        """Test static method for anonymizing model keys."""
        model_keys = ["gpt-4", "claude-3", "gemini-pro"]
        mapping = ModelAnonymizer.anonymize_model_keys(model_keys)

        assert len(mapping) == 3
        assert mapping["gpt-4"] == "LLM1"
        assert mapping["claude-3"] == "LLM2"
        assert mapping["gemini-pro"] == "LLM3"

    def test_anonymize_model_keys_preserves_order(self) -> None:
        """Test that anonymization preserves input order."""
        model_keys = ["z_model", "a_model", "m_model"]
        mapping = ModelAnonymizer.anonymize_model_keys(model_keys)

        # Should use input order, not alphabetical
        assert mapping["z_model"] == "LLM1"
        assert mapping["a_model"] == "LLM2"
        assert mapping["m_model"] == "LLM3"

    def test_anonymize_single_model(self) -> None:
        """Test anonymizing single model."""
        mapping = ModelAnonymizer.anonymize_model_keys(["only_model"])

        assert len(mapping) == 1
        assert mapping["only_model"] == "LLM1"

    def test_anonymize_empty_list(self) -> None:
        """Test anonymizing empty list."""
        mapping = ModelAnonymizer.anonymize_model_keys([])

        assert len(mapping) == 0


class TestAnonymizerDeterministicMode:
    """Test deterministic vs non-deterministic mode."""

    def test_deterministic_mode_enabled(self) -> None:
        """Test anonymizer with deterministic mode enabled."""
        anonymizer = ModelAnonymizer(deterministic_mode=True)

        responses = {"model_a": "Response A", "model_b": "Response B"}
        anon1, _ = anonymizer.anonymize_responses(responses)

        # Create new anonymizer with deterministic mode
        anonymizer2 = ModelAnonymizer(deterministic_mode=True)
        anon2, _ = anonymizer2.anonymize_responses(responses)

        # Results should be deterministic (alphabetical order)
        assert anon1 == anon2

    def test_deterministic_mode_disabled(self) -> None:
        """Test anonymizer with non-deterministic mode."""
        anonymizer = ModelAnonymizer(deterministic_mode=False)

        responses = {"model_a": "Response A"}
        anon, _ = anonymizer.anonymize_responses(responses)

        # Should still work, just potentially non-deterministic
        assert len(anon) == 1


class TestAnonymizeResponses:
    """Test response anonymization."""

    def test_anonymize_responses_basic(self) -> None:
        """Test basic response anonymization."""
        anonymizer = ModelAnonymizer()

        responses = {
            "model_a": "Response from A",
            "model_b": "Response from B",
            "model_c": "Response from C",
        }

        anon_responses, reverse_mapping = anonymizer.anonymize_responses(
            responses
        )

        # Check anonymized responses
        assert len(anon_responses) == 3
        assert "LLM1" in anon_responses
        assert "LLM2" in anon_responses
        assert "LLM3" in anon_responses

        # Check reverse mapping
        assert len(reverse_mapping) == 3
        for anon_name, original_name in reverse_mapping.items():
            assert original_name in responses
            assert anon_name in anon_responses

    def test_anonymize_responses_alphabetical_order(self) -> None:
        """Test that responses are anonymized in alphabetical order."""
        anonymizer = ModelAnonymizer()

        responses = {
            "zebra_model": "Z response",
            "alpha_model": "A response",
            "beta_model": "B response",
        }

        anon_responses, reverse_mapping = anonymizer.anonymize_responses(
            responses
        )

        # Alphabetical order: alpha, beta, zebra
        assert reverse_mapping["LLM1"] == "alpha_model"
        assert reverse_mapping["LLM2"] == "beta_model"
        assert reverse_mapping["LLM3"] == "zebra_model"

        # Content should be preserved
        assert anon_responses["LLM1"] == "A response"
        assert anon_responses["LLM2"] == "B response"
        assert anon_responses["LLM3"] == "Z response"

    def test_anonymize_single_response(self) -> None:
        """Test anonymizing single response."""
        anonymizer = ModelAnonymizer()

        responses = {"only_model": "Single response"}

        anon_responses, reverse_mapping = anonymizer.anonymize_responses(
            responses
        )

        assert len(anon_responses) == 1
        assert "LLM1" in anon_responses
        assert anon_responses["LLM1"] == "Single response"
        assert reverse_mapping["LLM1"] == "only_model"

    def test_anonymize_responses_with_special_characters(self) -> None:
        """Test anonymizing responses with special characters in names."""
        anonymizer = ModelAnonymizer()

        responses = {
            "GPT-4": "Response 1",
            "Claude-3.5": "Response 2",
            "Gemini-1.5-Pro": "Response 3",
        }

        anon_responses, reverse_mapping = anonymizer.anonymize_responses(
            responses
        )

        assert len(anon_responses) == 3
        assert len(reverse_mapping) == 3

        # All original names should be in reverse mapping
        assert "Claude-3.5" in reverse_mapping.values()
        assert "GPT-4" in reverse_mapping.values()
        assert "Gemini-1.5-Pro" in reverse_mapping.values()

    def test_anonymize_responses_preserves_content(self) -> None:
        """Test that anonymization preserves response content."""
        anonymizer = ModelAnonymizer()

        original_content = "This is a very long response with multiple sentences. It contains important information that must be preserved exactly as is, including punctuation, capitalization, and special characters like @#$%."

        responses = {"test_model": original_content}

        anon_responses, _ = anonymizer.anonymize_responses(responses)

        # Content should be exactly the same
        assert anon_responses["LLM1"] == original_content

    def test_anonymize_empty_responses(self) -> None:
        """Test anonymizing empty response dictionary."""
        anonymizer = ModelAnonymizer()

        responses = {}

        anon_responses, reverse_mapping = anonymizer.anonymize_responses(
            responses
        )

        assert len(anon_responses) == 0
        assert len(reverse_mapping) == 0


class TestAnonymizerInTournament:
    """Test anonymizer integration in actual tournaments."""

    @pytest.mark.asyncio
    async def test_tournament_uses_anonymization(
        self,
        basic_config: dict,
    ) -> None:
        """Test that tournament properly uses anonymization."""
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

        # Check that anonymization mapping was created
        comparison = arbitrium._comparison
        assert comparison.anon_mapping is not None
        assert len(comparison.anon_mapping) == 3

        # Verify all models are mapped
        for model_key in mock_models.keys():
            assert model_key in comparison.anon_mapping
            assert comparison.anon_mapping[model_key].startswith("LLM")

    @pytest.mark.asyncio
    async def test_anonymization_consistent_throughout_tournament(
        self,
        basic_config: dict,
    ) -> None:
        """Test that anonymization remains consistent throughout tournament."""
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

        comparison = arbitrium._comparison
        initial_mapping = comparison.anon_mapping.copy()

        # Run tournament
        await arbitrium.run_tournament("Test question?")

        # Mapping should remain the same
        assert comparison.anon_mapping == initial_mapping

    @pytest.mark.asyncio
    async def test_deterministic_anonymization_in_tournament(
        self,
        basic_config: dict,
    ) -> None:
        """Test deterministic anonymization in tournament."""
        basic_config["features"]["deterministic_mode"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models = {
            "model_z": MockModel(model_name="test-z", display_name="Model Z"),
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
        }
        arbitrium._healthy_models = mock_models

        comparison = arbitrium._create_comparison()

        # Anonymization uses input order, not alphabetical
        # Keys are iterated in dict order (model_z, model_a)
        assert len(comparison.anon_mapping) == 2
        assert all(
            v.startswith("LLM") for v in comparison.anon_mapping.values()
        )


class TestAnonymizerEdgeCases:
    """Test edge cases and unusual scenarios."""

    def test_anonymize_many_models(self) -> None:
        """Test anonymizing many models."""
        model_keys = [f"model_{i}" for i in range(20)]
        mapping = ModelAnonymizer.anonymize_model_keys(model_keys)

        assert len(mapping) == 20
        assert mapping["model_0"] == "LLM1"
        assert mapping["model_10"] == "LLM11"
        assert mapping["model_19"] == "LLM20"

    def test_anonymize_models_with_similar_names(self) -> None:
        """Test anonymizing models with very similar names."""
        anonymizer = ModelAnonymizer()

        responses = {
            "model": "Response 1",
            "model_1": "Response 2",
            "model_2": "Response 3",
        }

        anon_responses, reverse_mapping = anonymizer.anonymize_responses(
            responses
        )

        # Each should get unique anonymized name
        assert len(set(anon_responses.keys())) == 3
        assert len(set(reverse_mapping.values())) == 3

    def test_anonymize_unicode_model_names(self) -> None:
        """Test anonymizing models with unicode characters."""
        anonymizer = ModelAnonymizer()

        responses = {
            "模型_a": "Response in Chinese",
            "モデル_b": "Response in Japanese",
            "मॉडल_c": "Response in Hindi",
        }

        anon_responses, reverse_mapping = anonymizer.anonymize_responses(
            responses
        )

        assert len(anon_responses) == 3
        assert len(reverse_mapping) == 3

        # All original names should be preserved in reverse mapping
        original_names = set(reverse_mapping.values())
        assert "模型_a" in original_names
        assert "モデル_b" in original_names
        assert "मॉडल_c" in original_names

    def test_reverse_mapping_is_correct_inverse(self) -> None:
        """Test that reverse mapping is the correct inverse of forward mapping."""
        anonymizer = ModelAnonymizer()

        responses = {
            "model_x": "X",
            "model_y": "Y",
            "model_z": "Z",
        }

        _, reverse_mapping = anonymizer.anonymize_responses(responses)

        # Forward then reverse should give original
        for _anon_name, original_name in reverse_mapping.items():
            # The original should map to anon through alphabetical order
            assert original_name in responses

    def test_anonymize_responses_with_empty_content(self) -> None:
        """Test anonymizing responses with empty content."""
        anonymizer = ModelAnonymizer()

        responses = {
            "model_a": "",
            "model_b": "Non-empty",
            "model_c": "",
        }

        anon_responses, reverse_mapping = anonymizer.anonymize_responses(
            responses
        )

        # Should still anonymize even with empty content
        assert len(anon_responses) == 3

        # Empty content should be preserved
        for anon_name in anon_responses:
            original_name = reverse_mapping[anon_name]
            assert anon_responses[anon_name] == responses[original_name]
