"""Test provenance generation with complete data."""

import json
import tempfile
from pathlib import Path

import pytest

from arbitrium_core.domain.reporting.provenance import ProvenanceReport


@pytest.mark.asyncio
async def test_provenance_json_structure_complete():
    """Test that provenance JSON has all required fields populated."""
    # Create sample tournament data
    tournament_data = {
        "complete_tournament_history": {
            "Phase 1: Initial Answers": {
                "LLM1": "Initial answer from LLM1",
                "LLM2": "Initial answer from LLM2",
                "LLM3": "Initial answer from LLM3",
            },
            "Phase 2: Collaborative Analysis": {
                "criticisms": None,
                "feedback": None,
                "improved_answers": {
                    "LLM1": "Improved answer from LLM1",
                    "LLM2": "Improved answer from LLM2",
                    "LLM3": "Improved answer from LLM3",
                },
            },
            "Elimination Round 1": {
                "scores": {
                    "LLM1": {"LLM1": 8.0, "LLM2": 7.0, "LLM3": 6.0},
                    "LLM2": {"LLM1": 7.5, "LLM2": 8.5, "LLM3": 7.0},
                    "LLM3": {"LLM1": 6.5, "LLM2": 7.5, "LLM3": 8.0},
                },
                "evaluations": {
                    "LLM1": "LLM1 evaluation text",
                    "LLM2": "LLM2 evaluation text",
                    "LLM3": "LLM3 evaluation text",
                },
                "refined_answers": {
                    "LLM1": "Refined answer from LLM1",
                    "LLM2": "Refined answer from LLM2",
                },
            },
        },
        "eliminated_models": [
            {
                "model": "LLM3",
                "score": 6.5,
                "reason": "Lowest score",
                "insights_preserved": ["Insight 1", "Insight 2"],
            }
        ],
        "cost_summary": {
            "total_cost": "$0.1234",
        },
    }

    # Create provenance report
    report = ProvenanceReport(
        question="What is 2+2?",
        champion_model="LLM1",
        champion_answer="The answer is 4",
        tournament_data=tournament_data,
    )

    # Generate provenance metadata
    provenance = report._generate_provenance_metadata()

    # Verify structure
    assert "tournament_id" in provenance
    assert "question" in provenance
    assert "champion_model" in provenance
    assert "final_answer" in provenance
    assert "phases" in provenance
    assert "eliminations" in provenance
    assert "cost_summary" in provenance
    assert "generated_at" in provenance

    # Verify phases structure
    assert len(provenance["phases"]) == 3

    # Phase 1
    phase1 = provenance["phases"][0]
    assert phase1["type"] == "initial"
    assert "responses" in phase1
    assert "LLM1" in phase1["responses"]

    # Phase 2 - collaborative mode
    phase2 = provenance["phases"][1]
    assert phase2["type"] == "improvement"
    assert phase2["strategy"] == "collaborative"
    # In collaborative mode, only improved_answers should be present (no null fields!)
    assert "improved_answers" in phase2
    assert phase2["improved_answers"] is not None
    assert "LLM1" in phase2["improved_answers"]
    # criticisms and feedback should NOT exist in collaborative mode
    assert "criticisms" not in phase2
    assert "feedback" not in phase2

    # Phase 3
    phase3 = provenance["phases"][2]
    assert phase3["type"] == "evaluation"
    assert "scores" in phase3
    assert "evaluations" in phase3
    assert "refined_answers" in phase3
    # These should not be missing
    assert phase3["scores"] is not None
    assert "LLM1" in phase3["scores"]

    # Verify eliminations
    assert len(provenance["eliminations"]) == 1
    elim = provenance["eliminations"][0]
    assert elim["model"] == "LLM3"
    assert elim["score"] == 6.5
    assert len(elim["insights_preserved"]) == 2

    # Verify JSON serializability
    json_str = json.dumps(provenance, indent=2, ensure_ascii=False)
    assert json_str  # Should not fail
    parsed = json.loads(json_str)
    assert parsed["question"] == "What is 2+2?"


@pytest.mark.asyncio
async def test_provenance_handles_special_characters():
    """Test that provenance properly escapes special characters."""
    # Create data with special characters
    tournament_data = {
        "complete_tournament_history": {
            "Phase 1: Initial Answers": {
                "LLM1": 'Answer with "quotes" and \n newlines',
                "LLM2": "Answer with 'single quotes' and \t tabs",
            },
            "Phase 2: Collaborative Analysis": {
                "criticisms": None,
                "feedback": None,
                "improved_answers": {
                    "LLM1": "Improved with {braces} and [brackets]",
                    "LLM2": "Improved with $special @chars #hash",
                },
            },
        },
        "eliminated_models": [],
        "cost_summary": {},
    }

    report = ProvenanceReport(
        question='Question with "quotes"?',
        champion_model="LLM1",
        champion_answer='Answer with \n newlines and "quotes"',
        tournament_data=tournament_data,
    )

    provenance = report._generate_provenance_metadata()

    # Should be able to serialize to JSON without errors
    json_str = json.dumps(provenance, indent=2, ensure_ascii=False)
    assert json_str

    # Should be able to parse back
    parsed = json.loads(json_str)
    assert "quotes" in parsed["question"]
    assert "newlines" in parsed["final_answer"]


@pytest.mark.asyncio
async def test_provenance_saves_to_files():
    """Test that provenance saves all files correctly."""
    tournament_data = {
        "complete_tournament_history": {
            "Phase 1: Initial Answers": {
                "LLM1": "Answer 1",
                "LLM2": "Answer 2",
            },
            "Phase 2: Collaborative Analysis": {
                "criticisms": None,
                "feedback": None,
                "improved_answers": {
                    "LLM1": "Improved 1",
                    "LLM2": "Improved 2",
                },
            },
        },
        "eliminated_models": [],
        "cost_summary": {},
    }

    report = ProvenanceReport(
        question="Test question",
        champion_model="LLM1",
        champion_answer="Test answer",
        tournament_data=tournament_data,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        timestamp = "20250118_120000"
        saved_files = await report.save_to_file(tmpdir, timestamp)

        # Verify all files were created
        assert "champion_md" in saved_files
        assert "provenance_json" in saved_files
        assert "complete_history_json" in saved_files

        # Verify files exist
        assert Path(saved_files["champion_md"]).exists()
        assert Path(saved_files["provenance_json"]).exists()
        assert Path(saved_files["complete_history_json"]).exists()

        # Verify provenance JSON is valid
        provenance_path = Path(saved_files["provenance_json"])
        with open(provenance_path, encoding="utf-8") as f:
            provenance = json.load(f)

        # Should have all required fields
        assert provenance["question"] == "Test question"
        assert provenance["champion_model"] == "LLM1"
        assert provenance["final_answer"] == "Test answer"
        assert len(provenance["phases"]) == 2

        # Phase 2 should have improved_answers (collaborative mode has no null fields)
        phase2 = provenance["phases"][1]
        assert "improved_answers" in phase2
        assert phase2["improved_answers"] is not None
        # In collaborative mode, criticisms and feedback should NOT be present
        assert "criticisms" not in phase2
        assert "feedback" not in phase2
