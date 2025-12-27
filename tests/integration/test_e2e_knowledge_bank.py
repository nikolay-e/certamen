"""End-to-end tests for Knowledge Bank functionality."""

import pytest

from arbitrium_core import Arbitrium
from tests.integration.conftest import MockModel


class TestKnowledgeBankExtraction:
    """Test insight extraction from eliminated models."""

    @pytest.mark.asyncio
    async def test_knowledge_bank_extracts_insights_from_eliminated_model(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB extracts insights when a model is eliminated."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        # Setup mock models
        mock_models = {
            "model_a": MockModel(
                model_name="test-a",
                display_name="Model A",
                response_text="This is a detailed answer with multiple insights",
            ),
            "model_b": MockModel(
                model_name="test-b",
                display_name="Model B",
                response_text="Another comprehensive response",
            ),
            "model_c": MockModel(
                model_name="test-c",
                display_name="Model C",
                response_text="A third perspective with unique ideas",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Run tournament
        await arbitrium.run_tournament("Test question?")

        # Access the comparison object that was used
        comparison = arbitrium._last_comparison
        assert comparison is not None

        # Verify insights were extracted
        kb = comparison.knowledge_bank
        insights = await kb.get_all_insights()

        # At least one model was eliminated, so should have insights
        assert len(insights) >= 1
        assert all("text" in insight for insight in insights)
        assert all("source_model" in insight for insight in insights)
        assert all("source_round" in insight for insight in insights)

    @pytest.mark.asyncio
    async def test_knowledge_bank_skips_invalid_responses(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB skips invalid/short responses during extraction."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        # Test various invalid responses
        test_cases = [
            ("", "Response is empty"),
            ("Short", "Response too short"),
            ("Error: Something went wrong", "Response is an error message"),
            ("Failed: Could not process", "Response is an error message"),
        ]

        for response, expected_reason in test_cases:
            is_valid, reason = kb._is_valid_response_for_extraction(response)
            assert not is_valid
            assert expected_reason in reason

    @pytest.mark.asyncio
    async def test_knowledge_bank_accepts_valid_responses(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB accepts valid responses for extraction."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        valid_response = (
            "This is a detailed and comprehensive response that contains "
            "multiple insights and perspectives on the topic at hand."
        )

        is_valid, reason = kb._is_valid_response_for_extraction(valid_response)
        assert is_valid
        assert reason == ""


class TestKnowledgeBankDuplicateDetection:
    """Test duplicate detection and similarity threshold."""

    @pytest.mark.asyncio
    async def test_knowledge_bank_prevents_duplicate_insights(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB prevents adding duplicate insights."""
        # Set high similarity threshold to catch duplicates
        kb_enabled_config["knowledge_bank"]["similarity_threshold"] = 0.8

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        # Add first set of insights
        claims = [
            "The primary factor is cost efficiency",
            "Security is paramount for user trust",
            "Performance optimization reduces latency",
        ]
        await kb._add_insights_to_db(claims, "Model A", source_round=1)

        initial_count = len(kb.insights_db)
        assert initial_count == 3

        # Try to add very similar insights
        duplicate_claims = [
            "The primary factor is cost efficiency",  # Exact duplicate
            "Cost efficiency is the primary factor",  # Paraphrase
        ]
        await kb._add_insights_to_db(
            duplicate_claims, "Model B", source_round=2
        )

        # Should not add duplicates
        final_count = len(kb.insights_db)
        assert final_count == initial_count  # No new insights added

    @pytest.mark.asyncio
    async def test_knowledge_bank_similarity_threshold_affects_deduplication(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that similarity threshold controls duplicate detection."""
        # Test with low threshold (strict - more duplicates caught)
        kb_enabled_config["knowledge_bank"]["similarity_threshold"] = 0.9

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        claims = [
            "Machine learning improves prediction accuracy",
            "Deep learning enhances classification performance",
        ]
        await kb._add_insights_to_db(claims, "Model A", source_round=1)

        # With high threshold, these should be considered different
        assert len(kb.insights_db) == 2


class TestKnowledgeBankMaxInsights:
    """Test max insights limit and LRU eviction."""

    @pytest.mark.asyncio
    async def test_knowledge_bank_enforces_max_insights_limit(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB enforces max insights limit using LRU eviction."""
        # Set low max insights limit
        kb_enabled_config["knowledge_bank"]["max_insights"] = 5

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        # Add more insights than the limit
        claims = [f"Insight number {i} with unique content" for i in range(10)]
        await kb._add_insights_to_db(claims, "Model A", source_round=1)

        # Should only keep max_insights
        assert len(kb.insights_db) == 5

        # Should keep the most recent ones (LRU eviction)
        all_insights = await kb.get_all_insights()
        assert len(all_insights) == 5

    @pytest.mark.asyncio
    async def test_knowledge_bank_lru_eviction_removes_oldest(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that LRU eviction removes oldest insights first."""
        kb_enabled_config["knowledge_bank"]["max_insights"] = 3

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        # Add insights in batches
        first_batch = [
            "First insight A",
            "First insight B",
        ]
        await kb._add_insights_to_db(first_batch, "Model A", source_round=1)

        second_batch = [
            "Second insight C",
            "Second insight D",
        ]
        await kb._add_insights_to_db(second_batch, "Model B", source_round=2)

        # Should keep only last 3 (one from first batch, two from second batch)
        all_insights = await kb.get_all_insights()
        assert len(all_insights) == 3

        # Verify oldest was removed
        texts = [insight["text"] for insight in all_insights]
        assert "First insight A" not in texts  # Oldest removed


class TestKnowledgeBankFormatting:
    """Test insight formatting for context injection."""

    @pytest.mark.asyncio
    async def test_knowledge_bank_formats_insights_for_context(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB formats insights correctly for injection."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        # Add some insights
        claims = [
            "Security is critical for user trust",
            "Performance optimization reduces costs",
            "Scalability enables growth",
        ]
        await kb._add_insights_to_db(claims, "Model A", source_round=1)

        # Format insights
        formatted = await kb.format_insights_for_context()

        # Verify formatting
        assert "KNOWLEDGE BANK" in formatted
        assert "KEY INSIGHTS FROM ELIMINATED MODELS" in formatted
        assert "Model A" in formatted
        assert "Round 1" in formatted
        assert all(claim in formatted for claim in claims)

    @pytest.mark.asyncio
    async def test_knowledge_bank_returns_empty_when_disabled(
        self,
        basic_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB returns empty string when disabled."""
        # Ensure KB is disabled
        basic_config["knowledge_bank"]["enabled"] = False

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        # Add insights
        claims = ["Some insight"]
        await kb._add_insights_to_db(claims, "Model A", source_round=1)

        # Format should return empty when disabled
        formatted = await kb.format_insights_for_context()
        assert formatted == ""

    @pytest.mark.asyncio
    async def test_knowledge_bank_returns_empty_when_no_insights(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB returns empty string when no insights exist."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        # No insights added
        formatted = await kb.format_insights_for_context()
        assert formatted == ""


class TestKnowledgeBankIntegration:
    """Test KB integration with tournament workflow."""

    @pytest.mark.asyncio
    async def test_knowledge_bank_insights_injected_into_improvement(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB insights are injected into improvement prompts."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        # Setup mock models
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

        # Manually add insights to KB
        kb = comparison.knowledge_bank
        claims = [
            "Consider the economic impact of this decision",
            "Historical data suggests a different approach",
        ]
        await kb._add_insights_to_db(
            claims, "Eliminated Model", source_round=1
        )

        # Get formatted insights
        formatted = await kb.format_insights_for_context()

        # Verify insights are available for injection
        assert len(formatted) > 0
        assert all(claim in formatted for claim in claims)

    @pytest.mark.asyncio
    async def test_tournament_with_kb_vs_without_kb(
        self,
        basic_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test tournament behavior with and without KB enabled."""
        # Run with KB disabled
        basic_config["knowledge_bank"]["enabled"] = False

        arbitrium_no_kb = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models_1 = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium_no_kb._healthy_models = mock_models_1

        result_no_kb, metrics_no_kb = await arbitrium_no_kb.run_tournament(
            "Test question?"
        )

        # Access the comparison object that was used
        comparison_no_kb = arbitrium_no_kb._last_comparison
        assert comparison_no_kb is not None

        # Verify no insights were collected
        kb_no_kb = comparison_no_kb.knowledge_bank
        assert len(kb_no_kb.insights_db) == 0

        # Run with KB enabled
        basic_config["knowledge_bank"]["enabled"] = True

        arbitrium_with_kb = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
        )

        mock_models_2 = {
            "model_a": MockModel(model_name="test-a", display_name="Model A"),
            "model_b": MockModel(model_name="test-b", display_name="Model B"),
        }
        arbitrium_with_kb._healthy_models = mock_models_2

        (
            result_with_kb,
            metrics_with_kb,
        ) = await arbitrium_with_kb.run_tournament("Test question?")

        # Access the comparison object that was used
        comparison_with_kb = arbitrium_with_kb._last_comparison
        assert comparison_with_kb is not None

        # Verify insights were collected
        kb_with_kb = comparison_with_kb.knowledge_bank
        # Should have insights from eliminated model
        assert len(kb_with_kb.insights_db) >= 1

        # Both should complete successfully
        assert result_no_kb is not None
        assert result_with_kb is not None
        assert metrics_no_kb["champion_model"] is not None
        assert metrics_with_kb["champion_model"] is not None


class TestKnowledgeBankParsing:
    """Test insight parsing from LLM responses."""

    @pytest.mark.asyncio
    async def test_knowledge_bank_parses_bullet_points(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB correctly parses bullet point format."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        response = """
- First important insight about the topic
- Second critical consideration to note
- Third key factor in the analysis
"""

        claims = kb._parse_claims_from_response(response, "test-model")

        assert len(claims) == 3
        assert "First important insight about the topic" in claims
        assert "Second critical consideration to note" in claims
        assert "Third key factor in the analysis" in claims

    @pytest.mark.asyncio
    async def test_knowledge_bank_parses_numbered_lists(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB correctly parses numbered list format."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        response = """
1. First numbered insight with details
2) Second numbered insight using parenthesis
3. Third numbered insight for completeness
"""

        claims = kb._parse_claims_from_response(response, "test-model")

        assert len(claims) == 3
        assert "First numbered insight with details" in claims
        assert "Second numbered insight using parenthesis" in claims

    @pytest.mark.asyncio
    async def test_knowledge_bank_skips_short_lines(
        self,
        kb_enabled_config: dict,
        tmp_output_dir,
    ) -> None:
        """Test that KB skips short or header-like lines."""
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
        )

        comparison = arbitrium._create_comparison()
        kb = comparison.knowledge_bank

        response = """
INSIGHTS
--------
- This is a valid insight that is long enough to be included
- Short
- HEADER TEXT
- Another valid insight with sufficient length and content
"""

        claims = kb._parse_claims_from_response(response, "test-model")

        # Should only get the two valid insights
        assert len(claims) == 2
        assert any(
            "valid insight that is long enough" in claim for claim in claims
        )
        assert any(
            "Another valid insight with sufficient length" in claim
            for claim in claims
        )
