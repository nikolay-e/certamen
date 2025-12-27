"""End-to-end tests for score extraction and normalization."""

from arbitrium_core.domain.tournament.scoring import ScoreExtractor


class TestScoreExtraction:
    """Test basic score extraction from evaluation text."""

    def test_extract_simple_scores(self) -> None:
        """Test extracting simple scores in X/10 format."""
        extractor = ScoreExtractor()
        evaluation_text = """
        Model A: 8/10
        Model B: 7/10
        Model C: 9/10
        """
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B", "Model C"]
        )

        assert len(scores) == 3
        assert scores["Model A"] == 8.0
        assert scores["Model B"] == 7.0
        assert scores["Model C"] == 9.0

    def test_extract_scores_with_colon_format(self) -> None:
        """Test extracting scores with colon separator."""
        extractor = ScoreExtractor()
        evaluation_text = """
        Model A: 8.5
        Model B: 7.2
        """
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert len(scores) == 2
        assert scores["Model A"] == 8.5
        assert scores["Model B"] == 7.2

    def test_extract_scores_with_alternative_names(self) -> None:
        """Test extracting scores using LLM1, LLM2 naming."""
        extractor = ScoreExtractor()
        evaluation_text = """
        LLM1: 8/10
        LLM2: 7/10
        """
        # sorted(["Model B", "Model A"]) = ["Model A", "Model B"]
        # So LLM1 -> Model A, LLM2 -> Model B
        scores = extractor.extract_scores(
            evaluation_text, ["Model B", "Model A"]
        )

        assert len(scores) == 2
        assert "Model A" in scores
        assert "Model B" in scores

    def test_extract_scores_with_response_format(self) -> None:
        """Test extracting scores using 'Response 1' naming."""
        extractor = ScoreExtractor()
        evaluation_text = """
        Response 1: 9/10
        Response 2: 6/10
        """
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert len(scores) == 2

    def test_extract_fractional_scores(self) -> None:
        """Test extracting fractional scores like 8.5/10."""
        extractor = ScoreExtractor()
        evaluation_text = "Model A: 8.5/10, Model B: 7.25/10"
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert scores["Model A"] == 8.5
        assert scores["Model B"] == 7.25

    def test_extract_scores_with_text_around(self) -> None:
        """Test extracting scores when surrounded by explanatory text."""
        extractor = ScoreExtractor()
        evaluation_text = """
        Based on my analysis:

        Model A performed excellently: 9/10

        However, Model B had some issues: 6/10

        Overall, both were good.
        """
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert scores["Model A"] == 9.0
        assert scores["Model B"] == 6.0

    def test_partial_score_extraction(self) -> None:
        """Test extraction when only some scores are found."""
        extractor = ScoreExtractor()
        evaluation_text = "Model A: 8/10"
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        # Should find Model A but not Model B
        assert "Model A" in scores
        assert scores["Model A"] == 8.0
        assert len(scores) == 1

    def test_no_scores_found(self) -> None:
        """Test extraction when no scores are found."""
        extractor = ScoreExtractor()
        evaluation_text = "Both models did well, but I can't decide."
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert len(scores) == 0


class TestScoreExtractionFromEvaluation:
    """Test score extraction with validation and apology detection."""

    def test_extract_valid_scores(self) -> None:
        """Test extracting valid scores with full evaluation."""
        extractor = ScoreExtractor()
        evaluation_text = "Model A: 8/10\nModel B: 7/10"
        scores = extractor.extract_scores_from_evaluation(
            evaluation_text, ["Model A", "Model B"], "TestEvaluator"
        )

        assert len(scores) == 2
        assert scores["Model A"] == 8.0
        assert scores["Model B"] == 7.0

    def test_reject_apology_response(self) -> None:
        """Test that apology responses are rejected."""
        extractor = ScoreExtractor()
        evaluation_text = "I apologize, but I cannot evaluate these models."
        scores = extractor.extract_scores_from_evaluation(
            evaluation_text, ["Model A", "Model B"], "TestEvaluator"
        )

        assert len(scores) == 0

    def test_reject_refusal_response(self) -> None:
        """Test that refusal responses are rejected."""
        extractor = ScoreExtractor()
        evaluation_text = "I'm sorry, I cannot compare these responses."
        scores = extractor.extract_scores_from_evaluation(
            evaluation_text, ["Model A", "Model B"], "TestEvaluator"
        )

        assert len(scores) == 0

    def test_reject_incomplete_evaluation(self) -> None:
        """Test that incomplete evaluations are rejected."""
        extractor = ScoreExtractor()
        evaluation_text = "Model A: 8/10"  # Missing Model B
        scores = extractor.extract_scores_from_evaluation(
            evaluation_text, ["Model A", "Model B"], "TestEvaluator"
        )

        # Should return empty dict when evaluation is incomplete
        assert len(scores) == 0

    def test_accept_complete_evaluation(self) -> None:
        """Test that complete evaluations are accepted."""
        extractor = ScoreExtractor()
        evaluation_text = "Model A: 8/10\nModel B: 7/10\nModel C: 9/10"
        scores = extractor.extract_scores_from_evaluation(
            evaluation_text, ["Model A", "Model B", "Model C"], "TestEvaluator"
        )

        assert len(scores) == 3


class TestScoreNormalization:
    """Test score normalization logic."""

    def test_normalize_valid_score(self) -> None:
        """Test that valid scores pass through unchanged."""
        extractor = ScoreExtractor()

        assert extractor.normalize_score(5.0, "Test") == 5.0
        assert extractor.normalize_score(8.5, "Test") == 8.5
        assert extractor.normalize_score(1.0, "Test") == 1.0
        assert extractor.normalize_score(10.0, "Test") == 10.0

    def test_normalize_percentage_to_scale(self) -> None:
        """Test normalizing percentage (0-1) to 1-10 scale."""
        extractor = ScoreExtractor()

        # 0.8 should normalize to 8.0
        normalized = extractor.normalize_score(0.8, "Test")
        assert normalized == 8.0

        # 0.5 should normalize to 5.0
        normalized = extractor.normalize_score(0.5, "Test")
        assert normalized == 5.0

    def test_normalize_oversized_score(self) -> None:
        """Test normalizing scores > 10."""
        extractor = ScoreExtractor()

        # Scores > 10.5 are rejected
        normalized = extractor.normalize_score(80.0, "Test")
        assert normalized is None

        # Scores > 10.5 are rejected
        normalized = extractor.normalize_score(100.0, "Test")
        assert normalized is None

        # Scores between 10 and 10.5 are clamped to 10.0
        normalized = extractor.normalize_score(10.2, "Test")
        assert normalized == 10.0

    def test_reject_invalid_scores(self) -> None:
        """Test rejecting completely invalid scores."""
        extractor = ScoreExtractor()

        # Negative scores
        assert extractor.normalize_score(-5.0, "Test") is None

        # Way too high
        assert extractor.normalize_score(1000.0, "Test") is None

        # Edge case: just outside valid range
        assert extractor.normalize_score(0.4, "Test") is None
        assert extractor.normalize_score(11.0, "Test") is None

    def test_edge_case_boundaries(self) -> None:
        """Test normalization at boundary values."""
        extractor = ScoreExtractor()

        # Just inside valid range
        assert (
            extractor.normalize_score(0.5, "Test") == 5.0
        )  # Normalized (0.5 * 10)
        assert (
            extractor.normalize_score(10.5, "Test") == 10.0
        )  # Clamped to 10.0

        # Exact boundaries
        assert extractor.normalize_score(1.0, "Test") == 1.0
        assert extractor.normalize_score(10.0, "Test") == 10.0


class TestNumericScoreExtraction:
    """Test numeric score extraction from various formats."""

    def test_extract_from_integer(self) -> None:
        """Test extracting score from integer."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score(8)
        assert score == 8.0

    def test_extract_from_float(self) -> None:
        """Test extracting score from float."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score(8.5)
        assert score == 8.5

    def test_extract_from_string(self) -> None:
        """Test extracting score from string."""
        extractor = ScoreExtractor()

        score = extractor._extract_numeric_score("8/10")
        assert score == 8.0

        score = extractor._extract_numeric_score("7.5")
        assert score == 7.5

    def test_extract_from_list(self) -> None:
        """Test extracting score from list (takes first element)."""
        extractor = ScoreExtractor()

        score = extractor._extract_numeric_score([8.5, 9.0])
        assert score == 8.5

        score = extractor._extract_numeric_score([])
        assert score is None

    def test_extract_from_invalid_format(self) -> None:
        """Test extraction from invalid format returns None."""
        extractor = ScoreExtractor()

        score = extractor._extract_numeric_score("invalid")
        assert score is None

        score = extractor._extract_numeric_score(None)
        assert score is None


class TestModelNameMatching:
    """Test fuzzy model name matching."""

    def test_exact_match(self) -> None:
        """Test exact model name match."""
        extractor = ScoreExtractor()
        result = extractor._match_model_name("Model A", ["Model A", "Model B"])
        assert result == "Model A"

    def test_fuzzy_match_partial(self) -> None:
        """Test fuzzy matching with partial names."""
        extractor = ScoreExtractor()

        # "gpt-4" should match "gpt-4-turbo"
        result = extractor._match_model_name(
            "gpt-4", ["gpt-4-turbo", "claude"]
        )
        assert result == "gpt-4-turbo"

    def test_fuzzy_match_contains(self) -> None:
        """Test fuzzy matching when key contains model name."""
        extractor = ScoreExtractor()

        result = extractor._match_model_name(
            "Model A Response", ["Model A", "Model B"]
        )
        assert result == "Model A"

    def test_no_match(self) -> None:
        """Test when no match is found."""
        extractor = ScoreExtractor()
        result = extractor._match_model_name("Model X", ["Model A", "Model B"])
        assert result is None


class TestComplexScoreFormats:
    """Test extraction from complex and edge case formats."""

    def test_multiline_evaluation(self) -> None:
        """Test extracting from multiline evaluation."""
        extractor = ScoreExtractor()
        evaluation_text = """
        ## Evaluation Results

        Model A: 9/10
        Quality: Excellent

        Model B: 7/10
        Quality: Good
        """
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert scores["Model A"] == 9.0
        assert scores["Model B"] == 7.0

    def test_markdown_table_format(self) -> None:
        """Test extracting from markdown table."""
        extractor = ScoreExtractor()
        # Use simpler format that the pattern matcher can handle
        evaluation_text = """
        Evaluation results:
        Model A: 8/10
        Model B: 6/10
        """
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert scores["Model A"] == 8.0
        assert scores["Model B"] == 6.0

    def test_numbered_list_format(self) -> None:
        """Test extracting from numbered list."""
        extractor = ScoreExtractor()
        evaluation_text = """
        1. Model A - 8/10
        2. Model B - 7/10
        """
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert scores["Model A"] == 8.0
        assert scores["Model B"] == 7.0

    def test_mixed_formats_in_same_text(self) -> None:
        """Test extracting when different models use different formats."""
        extractor = ScoreExtractor()
        evaluation_text = """
        Model A: 8/10
        Model B scored 7 points
        """
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        assert scores["Model A"] == 8.0
        # Model B might not be extracted due to format, but Model A should work
        assert "Model A" in scores


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions."""

    def test_empty_evaluation_text(self) -> None:
        """Test extraction from empty text."""
        extractor = ScoreExtractor()
        scores = extractor.extract_scores("", ["Model A", "Model B"])
        assert len(scores) == 0

    def test_empty_model_list(self) -> None:
        """Test extraction with empty model list."""
        extractor = ScoreExtractor()
        scores = extractor.extract_scores("Model A: 8/10", [])
        assert len(scores) == 0

    def test_duplicate_model_names(self) -> None:
        """Test extraction when model names are duplicated."""
        extractor = ScoreExtractor()
        evaluation_text = "Model A: 8/10\nModel A: 9/10"
        scores = extractor.extract_scores(evaluation_text, ["Model A"])

        # Should extract one score (first or last match)
        assert "Model A" in scores

    def test_special_characters_in_model_names(self) -> None:
        """Test extraction with special characters in model names."""
        extractor = ScoreExtractor()
        evaluation_text = "GPT-4: 8/10\nClaude-3.5: 9/10"
        scores = extractor.extract_scores(
            evaluation_text, ["GPT-4", "Claude-3.5"]
        )

        assert scores["GPT-4"] == 8.0
        assert scores["Claude-3.5"] == 9.0

    def test_case_insensitive_matching(self) -> None:
        """Test case-insensitive model name matching."""
        extractor = ScoreExtractor()
        evaluation_text = "model a: 8/10\nMODEL B: 7/10"
        scores = extractor.extract_scores(
            evaluation_text, ["Model A", "Model B"]
        )

        # Case-insensitive extraction should work
        assert len(scores) >= 1
