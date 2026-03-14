from certamen_core.domain.disagreement.resolver import DisagreementInvestigator
from tests.integration.conftest import MockModel


def test_parse_resolution_resolved():
    status, conf = DisagreementInvestigator._parse_resolution(
        "After reviewing evidence, this is resolved. Confidence: 0.9"
    )
    assert status == "resolved"
    assert conf == 0.9


def test_parse_resolution_unresolved():
    status, _ = DisagreementInvestigator._parse_resolution(
        "The disagreement remains unresolved. No confidence value given."
    )
    assert status == "unresolved"


def test_parse_resolution_partially():
    status, _ = DisagreementInvestigator._parse_resolution(
        "This is partially_resolved based on current evidence."
    )
    assert status == "partially_resolved"


def test_find_model_exact_key():
    models = {"gpt4": MockModel(model_name="gpt4", display_name="GPT-4")}
    result = DisagreementInvestigator._find_model("gpt4", models)
    assert result is not None
    assert result.display_name == "GPT-4"


def test_find_model_display_name_exact():
    models = {"m1": MockModel(model_name="m1", display_name="Claude")}
    result = DisagreementInvestigator._find_model("Claude", models)
    assert result is not None


def test_find_model_prefix_match():
    models = {
        "m1": MockModel(
            model_name="m1", display_name="Claude (anthropic/claude-3)"
        )
    }
    result = DisagreementInvestigator._find_model("Claude", models)
    assert result is not None


def test_find_model_substring_match():
    models = {
        "m1": MockModel(model_name="m1", display_name="GPT-4o (openai/gpt-4o)")
    }
    result = DisagreementInvestigator._find_model("GPT", models)
    assert result is not None


def test_find_model_no_match_returns_none():
    models = {"m1": MockModel(model_name="m1", display_name="SomeModel")}
    result = DisagreementInvestigator._find_model(
        "nonexistent_xyz_abc", models
    )
    assert result is None


def test_parse_resolution_unresolved_not_matched_as_resolved():
    # Word-boundary check: "unresolved" should not match as "resolved"
    status, _ = DisagreementInvestigator._parse_resolution(
        "Resolution status: unresolved. Confidence: 0.1"
    )
    assert status == "unresolved"
