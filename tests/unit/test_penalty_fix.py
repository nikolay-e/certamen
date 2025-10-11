#!/usr/bin/env python3
"""Test the new penalty logic - invalid evaluations should be discarded."""


from arbitrium.core.comparison import ScoreExtractor
from arbitrium.logging import get_contextual_logger, setup_logging

setup_logging(enable_file_logging=False)
logger = get_contextual_logger("test")


def test_valid_evaluation_all_models_scored() -> None:
    """Test valid evaluation when all models are scored."""
    extractor = ScoreExtractor()

    evaluation_text = """
LLM1: 9/10
LLM2: 7/10
LLM3: 8/10
"""

    scores = extractor.extract_scores_from_evaluation(evaluation_text, ["LLM1", "LLM2", "LLM3"])
    assert scores == {"LLM1": 9.0, "LLM2": 7.0, "LLM3": 8.0}


def test_invalid_evaluation_missing_one_model() -> None:
    """Test invalid evaluation when one model is missing."""
    extractor = ScoreExtractor()

    evaluation_text = """
LLM1: 9/10
LLM3: 8/10
"""

    scores = extractor.extract_scores_from_evaluation(evaluation_text, ["LLM1", "LLM2", "LLM3"])
    assert scores == {}


def test_invalid_evaluation_missing_multiple_models() -> None:
    """Test invalid evaluation when multiple models are missing."""
    extractor = ScoreExtractor()

    evaluation_text = """
LLM1: 9/10
"""

    scores = extractor.extract_scores_from_evaluation(evaluation_text, ["LLM1", "LLM2", "LLM3"])
    assert scores == {}
