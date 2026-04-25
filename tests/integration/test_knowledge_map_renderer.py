from certamen.domain.knowledge_map.builder import (
    ConsensusItem,
    KnowledgeMap,
)
from certamen.domain.knowledge_map.renderer import KnowledgeMapRenderer


def _make_km() -> KnowledgeMap:
    return KnowledgeMap(
        question="Should we migrate to microservices?",
        consensus=[
            ConsensusItem(claim="Team size matters", confidence="HIGH"),
            ConsensusItem(claim="Cost will increase", confidence="MEDIUM"),
        ],
        unique_insights={"GPT-4": ["Strangler fig pattern reduces risk"]},
        known_unknowns=["How long will the migration take?"],
        assumptions=["Budget is not constrained"],
        confidence_distribution={
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 0,
            "UNCERTAIN": 0,
        },
        synthesis="Microservices migration requires careful planning.",
        champion_model="gpt4",
        exploration_branches=["What are the risks?", "What is the timeline?"],
    )


def test_to_markdown_has_sections():
    renderer = KnowledgeMapRenderer()
    km = _make_km()
    md = renderer.to_markdown(km)
    assert "# Knowledge Map" in md
    assert "## Synthesis" in md
    assert "## Consensus" in md
    assert "## Unique Insights" in md
    assert "## Known Unknowns" in md
    assert "## Assumptions" in md
    assert "## Suggested Follow-Up Questions" in md


def test_to_markdown_contains_content():
    renderer = KnowledgeMapRenderer()
    km = _make_km()
    md = renderer.to_markdown(km)
    assert "Should we migrate" in md
    assert "Microservices migration requires careful planning" in md
    assert "Team size matters" in md
    assert "Strangler fig pattern" in md


def test_to_json_has_keys():
    renderer = KnowledgeMapRenderer()
    km = _make_km()
    data = renderer.to_json(km)
    assert "question" in data
    assert "synthesis" in data
    assert "consensus" in data
    assert "disagreements" in data
    assert "unique_insights" in data
    assert "known_unknowns" in data
    assert "assumptions" in data
    assert "exploration_branches" in data


def test_to_json_consensus_structure():
    renderer = KnowledgeMapRenderer()
    km = _make_km()
    data = renderer.to_json(km)
    assert len(data["consensus"]) == 2
    first = data["consensus"][0]
    assert "claim" in first
    assert "confidence" in first


def test_to_markdown_empty_sections_hidden():
    renderer = KnowledgeMapRenderer()
    km = KnowledgeMap(
        question="Simple?",
        synthesis="Simple answer.",
        champion_model="m1",
    )
    md = renderer.to_markdown(km)
    assert "## Consensus" not in md
    assert "## Known Unknowns" not in md
