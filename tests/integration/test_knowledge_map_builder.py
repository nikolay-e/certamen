from certamen_core.domain.knowledge_map.builder import (
    KnowledgeMapBuilder,
)

_CONSENSUS_TEXT = """\
CONSENSUS: Microservices increase deployment flexibility [HIGH]
CONSENSUS: Team size is a critical factor [MEDIUM]
CONSENSUS: Cost will increase initially [LOW]
"""

_UNIQUE_INSIGHTS_TEXT = """\
MODEL: GPT-4
INSIGHT: The real bottleneck is organizational, not technical
---
MODEL: Claude
INSIGHT: A strangler fig pattern reduces migration risk
---
MODEL: GPT-4
INSIGHT: Database coupling is the hardest part of the migration
"""


def test_parse_consensus_finds_all():
    result = KnowledgeMapBuilder._parse_consensus(_CONSENSUS_TEXT)
    assert len(result) == 3


def test_parse_consensus_confidence_levels():
    result = KnowledgeMapBuilder._parse_consensus(_CONSENSUS_TEXT)
    confidences = [r.confidence for r in result]
    assert "HIGH" in confidences
    assert "MEDIUM" in confidences
    assert "LOW" in confidences


def test_parse_consensus_empty():
    result = KnowledgeMapBuilder._parse_consensus("No consensus here.")
    assert result == []


def test_parse_unique_insights_groups_by_model():
    result = KnowledgeMapBuilder._parse_unique_insights(_UNIQUE_INSIGHTS_TEXT)
    assert "GPT-4" in result
    assert "Claude" in result
    assert len(result["GPT-4"]) == 2
    assert len(result["Claude"]) == 1


def test_parse_unique_insights_empty():
    result = KnowledgeMapBuilder._parse_unique_insights("No insights here.")
    assert result == {}


def test_parse_questions_extracts_numbered():
    text = "1. What is the real bottleneck?\n2. How does the team scale?\n3. This is not a question"
    result = KnowledgeMapBuilder._parse_questions(text)
    assert len(result) == 2
    assert all("?" in q for q in result)
