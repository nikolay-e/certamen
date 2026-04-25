from certamen.domain.disagreement.detector import DisagreementDetector

_SAMPLE_TEXT = """\
DISAGREEMENT: Whether microservices reduce deployment time
POSITION_A: Model A: Microservices significantly reduce deployment time through independent deployability
POSITION_B: Model B: Microservices increase deployment complexity and often slow it down
SIGNIFICANCE: Core to the original question about migration feasibility
---
DISAGREEMENT: Team size requirements
POSITION_A: Model A: A team of 10 engineers is sufficient for microservices
POSITION_B: Model B: Microservices require specialized DevOps expertise beyond the current team
SIGNIFICANCE: Directly affects feasibility given 18-month runway
"""


def test_parse_disagreements_finds_two():
    result = DisagreementDetector._parse_disagreements(_SAMPLE_TEXT)
    assert len(result) == 2


def test_parse_disagreements_topic():
    result = DisagreementDetector._parse_disagreements(_SAMPLE_TEXT)
    topics = [d.topic for d in result]
    assert any("microservices" in t.lower() for t in topics)


def test_parse_disagreements_positions():
    result = DisagreementDetector._parse_disagreements(_SAMPLE_TEXT)
    first = result[0]
    assert len(first.positions) == 2


def test_parse_no_disagreements():
    result = DisagreementDetector._parse_disagreements("NO_DISAGREEMENTS")
    assert result == []


def test_parse_disagreements_empty_blocks():
    result = DisagreementDetector._parse_disagreements("")
    assert result == []
