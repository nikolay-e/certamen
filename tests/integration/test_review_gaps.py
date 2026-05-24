import pytest

from certamen.application.workflow.nodes.base import rank_by_scores
from certamen.domain.disagreement.resolver import DisagreementInvestigator
from certamen.domain.errors import ExceptionClassifier
from certamen.domain.tournament.anonymizer import ModelAnonymizer
from certamen.domain.tournament.scoring import ScoreExtractor


class _BadRequestError(Exception):
    pass


class TestExceptionClassifierRetryability:
    def test_real_rate_limit_messages_are_retryable(self) -> None:
        for msg in (
            "Rate limit exceeded",
            "429 Too Many Requests",
            "quota for this project has been used up",
        ):
            result = ExceptionClassifier.classify(RuntimeError(msg))
            assert result.error_type == "rate_limit"
            assert result.is_retryable is True

    def test_context_length_exceeded_is_not_retryable(self) -> None:
        # Regression: the bare "exceeded" rate-limit pattern used to mark a
        # permanent context-length error as retryable -> infinite retry loop.
        result = ExceptionClassifier.classify(
            RuntimeError(
                "This model's maximum context length is 8192 tokens; "
                "your messages exceeded it"
            )
        )
        assert result.error_type != "rate_limit"
        assert result.is_retryable is False

    def test_timeout_is_retryable(self) -> None:
        result = ExceptionClassifier.classify(
            RuntimeError("Request timed out")
        )
        assert result.error_type == "timeout"
        assert result.is_retryable is True

    def test_bad_request_with_docs_url_is_not_rate_limit(self) -> None:
        # Documented foot-gun: a BadRequestError carrying an https://docs link
        # must classify as a permanent bad_request, never a retryable rate_limit.
        exc = _BadRequestError(
            "LLM Provider NOT provided. "
            "See https://docs.litellm.ai/docs/providers"
        )
        result = ExceptionClassifier.classify(exc)
        assert result.error_type == "bad_request"
        assert result.is_retryable is False


class TestScoreExtractionGaps:
    def test_apology_returns_no_scores(self) -> None:
        extractor = ScoreExtractor()
        text = "I cannot evaluate these responses without more context."
        assert (
            extractor.extract_scores_from_evaluation(
                text, ["Model A", "Model B"], "judge"
            )
            == {}
        )

    def test_partial_scores_return_empty_to_avoid_unfair_penalty(self) -> None:
        extractor = ScoreExtractor()
        # Only Model A is scored; Model B is missing.
        text = "Model A: 8/10"
        assert (
            extractor.extract_scores_from_evaluation(
                text, ["Model A", "Model B"], "judge"
            )
            == {}
        )

    def test_complete_scores_are_extracted(self) -> None:
        extractor = ScoreExtractor()
        text = "Model A: 8/10\nModel B: 6/10"
        scores = extractor.extract_scores_from_evaluation(
            text, ["Model A", "Model B"], "judge"
        )
        assert scores == {"Model A": 8.0, "Model B": 6.0}


class TestModelAnonymizerRoundTrip:
    def test_reverse_mapping_recovers_original_names(self) -> None:
        anon = ModelAnonymizer()
        responses = {"gpt-4o": "answer-g", "claude": "answer-c"}
        anonymized, reverse = anon.anonymize_responses(responses)
        # Keys are sorted before labelling: ["claude", "gpt-4o"] -> LLM1, LLM2.
        assert reverse == {"LLM1": "claude", "LLM2": "gpt-4o"}
        # Every anon key maps back to a real model and preserves its content.
        for anon_key, text in anonymized.items():
            assert responses[reverse[anon_key]] == text

    def test_anonymized_keys_are_sequential_labels(self) -> None:
        anon = ModelAnonymizer()
        anonymized, _ = anon.anonymize_responses(
            {"a": "1", "b": "2", "c": "3"}
        )
        assert set(anonymized.keys()) == {"LLM1", "LLM2", "LLM3"}


class TestParseResolution:
    def test_integer_confidence_is_accepted(self) -> None:
        # Regression: "Confidence: 1" used to fall through to the 0.5 default.
        status, conf = DisagreementInvestigator._parse_resolution(
            "Status: resolved. Confidence: 1"
        )
        assert status == "resolved"
        assert conf == 1.0

    def test_zero_integer_confidence(self) -> None:
        _, conf = DisagreementInvestigator._parse_resolution(
            "unresolved. Confidence: 0"
        )
        assert conf == 0.0

    def test_decimal_confidence_still_works(self) -> None:
        status, conf = DisagreementInvestigator._parse_resolution(
            "partially resolved. Confidence: 0.7"
        )
        assert status == "partially_resolved"
        assert conf == pytest.approx(0.7)


class TestRankByScores:
    def test_ranks_by_score_descending(self) -> None:
        models = {"a": object(), "b": object(), "c": object()}
        scores = {"a": 5.0, "b": 9.0, "c": 7.0}
        assert rank_by_scores(models, scores) == ["b", "c", "a"]

    def test_missing_scores_default_to_zero_preserving_order(self) -> None:
        # Documents current behavior: unscored models sink to 0 and keep
        # insertion order (a deterministic, if blunt, tie-break).
        models = {"first": object(), "second": object()}
        assert rank_by_scores(models, {}) == ["first", "second"]
