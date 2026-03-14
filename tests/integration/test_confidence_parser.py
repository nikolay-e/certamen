from certamen_core.domain.confidence.parser import ConfidenceParser


def test_parse_confidence_tags():
    parser = ConfidenceParser()
    text = "The earth orbits the sun [HIGH] and dark matter exists [LOW]"
    claims = parser.parse_confidence_tags(text)
    assert len(claims) == 2
    assert claims[0].confidence == "HIGH"
    assert claims[1].confidence == "LOW"


def test_parse_confidence_tags_uncertain():
    parser = ConfidenceParser()
    text = "We don't fully understand consciousness [UNCERTAIN] but neurons fire [MEDIUM]"
    claims = parser.parse_confidence_tags(text)
    assert len(claims) == 2
    assert claims[0].confidence == "UNCERTAIN"
    assert claims[1].confidence == "MEDIUM"


def test_parse_confidence_tags_no_matches():
    parser = ConfidenceParser()
    claims = parser.parse_confidence_tags(
        "Plain text with no confidence tags."
    )
    assert claims == []


def test_extract_known_unknowns():
    parser = ConfidenceParser()
    text = "Some answer.\nKNOWN_UNKNOWNS:\n- How fast it spreads\n- Long-term effects"
    unknowns = parser.extract_known_unknowns(text)
    assert len(unknowns) == 2
    assert any("How fast" in u for u in unknowns)


def test_extract_known_unknowns_missing():
    parser = ConfidenceParser()
    assert parser.extract_known_unknowns("No unknowns here.") == []


def test_extract_assumptions():
    parser = ConfidenceParser()
    text = "Answer here.\nASSUMPTIONS:\n- Market stays stable\n- No regulation changes"
    assumptions = parser.extract_assumptions(text)
    assert len(assumptions) == 2
    assert any("Market" in a for a in assumptions)


def test_extract_assumptions_missing():
    parser = ConfidenceParser()
    assert parser.extract_assumptions("No assumptions here.") == []
