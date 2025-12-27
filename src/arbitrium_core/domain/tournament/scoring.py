"""Score extraction and normalization for tournament evaluation."""

import re

from arbitrium_core.shared.constants import (
    ANONYMOUS_MODEL_PREFIX,
    ANONYMOUS_RESPONSE_PREFIX,
    SCORE_EXTRACTION_PATTERNS,
)
from arbitrium_core.shared.logging import get_contextual_logger
from arbitrium_core.shared.validation.response import detect_apology_or_refusal


class ScoreExtractor:
    """Extracts and normalizes scores from LLM evaluation responses."""

    def __init__(self) -> None:
        self.logger = get_contextual_logger("arbitrium.scorer")

    def extract_scores(
        self, evaluation_text: str, model_names: list[str]
    ) -> dict[str, float]:
        # Try pattern matching with exact model names
        scores = self._extract_scores_using_pattern_matching(
            evaluation_text, model_names
        )

        # If we didn't get all scores, try alternative names (LLM1, Response 1, etc.)
        if len(scores) < len(model_names):
            self.logger.debug(
                f"Pattern matching found {len(scores)}/{len(model_names)} scores, trying alternative names"
            )
            alternative_scores = self._extract_scores_with_alternative_names(
                evaluation_text, model_names
            )
            scores.update(alternative_scores)

        if len(scores) >= len(model_names):
            self.logger.debug(
                f"Successfully extracted all {len(scores)} scores"
            )
        else:
            self.logger.warning(
                f"Found only {len(scores)}/{len(model_names)} scores"
            )

        return scores

    def extract_scores_from_evaluation(
        self,
        evaluation_text: str,
        model_names: list[str],
        evaluator_name: str = "Unknown",
    ) -> dict[str, float]:
        self.logger.debug(f"[{evaluator_name}] Parsing evaluation response")

        # Detect apology/refusal responses
        if detect_apology_or_refusal(evaluation_text):
            self.logger.error(
                f"[{evaluator_name}] Model returned apology/refusal instead of evaluation"
            )
            return {}

        scores = self.extract_scores(evaluation_text, model_names)
        missing_models = set(model_names) - set(scores.keys())

        if missing_models:
            self.logger.warning(
                f"[{evaluator_name}] Could not extract scores for {len(missing_models)} models: "
                f"{', '.join(sorted(missing_models))}. Evaluation may be incomplete."
            )
            # Return empty dict when evaluation is incomplete to avoid unfair penalties
            return {}

        return scores

    def normalize_score(self, score: float, evaluator: str) -> float | None:
        # Reject scores outside the valid range [0.5, 10.5]
        if score < 0.5 or score > 10.5:
            self.logger.error(
                f"Rejecting invalid score from {evaluator}: {score} (must be 1.0-10.0)"
            )
            return None

        # Normalize scores that are slightly out of bounds
        if score > 10:
            normalized = min(score, 10.0)
            self.logger.warning(
                f"Normalizing score from {evaluator}: {score} → {normalized:.2f}"
            )
            return normalized

        if 0 < score < 1:
            normalized = max(score * 10.0, 1.0)
            self.logger.warning(
                f"Normalizing score from {evaluator}: {score} → {normalized:.2f}"
            )
            return normalized

        return score

    # === Private Methods ===

    def _extract_numeric_score(self, score_value: object) -> float | None:
        if isinstance(score_value, list):
            if len(score_value) > 0:
                score_value = score_value[0]
            else:
                return None

        if isinstance(score_value, (int, float)):
            return float(score_value)

        score_str = str(score_value)
        patterns = [
            r"(\d+\.?\d*)\s*/\s*10",
            r"(\d+\.?\d*)",
        ]

        for i, pattern in enumerate(patterns):
            match = re.search(pattern, score_str)
            if match:
                try:
                    score = float(match.group(1))
                    self.logger.debug(
                        f"Extracted score {score} using pattern {i + 1}"
                    )
                    return score
                except (ValueError, IndexError) as e:
                    self.logger.debug(
                        f"Pattern {i + 1} matched but failed to parse: {e}"
                    )
                    continue

        self.logger.debug(
            f"Failed to extract numeric score from: {score_str[:50]}"
        )
        return None

    def _match_model_name(
        self, key: str, model_names: list[str]
    ) -> str | None:
        if key in model_names:
            return key

        for model_name in model_names:
            if model_name in key or key in model_name:
                self.logger.debug(f"Fuzzy matched '{key}' to '{model_name}'")
                return model_name

        return None

    def _try_extract_fractional_score(
        self,
        match: re.Match[str],
        model_name: str,
    ) -> float | None:
        try:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            score_value = (numerator / denominator) * 10.0
            self.logger.debug(
                f"Extracted fractional score {numerator}/{denominator} = {score_value}/10 for {model_name}"
            )
            return score_value
        except ValueError:
            # Group 1 is not a number, use group 2 as score
            try:
                score_value = float(match.group(2))
                self.logger.debug(
                    f"Extracted score {score_value} for {model_name}"
                )
                return score_value
            except ValueError:
                return None
        except ZeroDivisionError:
            return None

    def _extract_simple_score(
        self,
        match: re.Match[str],
        model_name: str,
    ) -> float | None:
        try:
            score_value = float(
                match.group(1) if len(match.groups()) == 1 else match.group(2)
            )
            self.logger.debug(
                f"Extracted score {score_value} for {model_name}"
            )
            return score_value
        except (ValueError, TypeError, IndexError):
            return None

    def _try_extract_score_from_match(
        self,
        match: re.Match[str],
        model_name: str,
    ) -> float | None:
        # Try fractional score if we have 2 groups
        if len(match.groups()) >= 2 and match.group(2):
            score = self._try_extract_fractional_score(match, model_name)
            if score is not None:
                return score

        # Fall back to simple score extraction
        return self._extract_simple_score(match, model_name)

    def _extract_score_for_model(
        self,
        evaluation_text: str,
        model_name: str,
        patterns: list[str],
    ) -> float | None:
        for pattern in patterns:
            formatted_pattern = pattern.format(
                model_name=re.escape(model_name)
            )
            match = re.search(
                formatted_pattern,
                evaluation_text,
                re.IGNORECASE | re.DOTALL | re.MULTILINE,
            )

            if match:
                score = self._try_extract_score_from_match(match, model_name)
                if score is not None:
                    return score
                # Continue to next pattern if this one failed
                self.logger.warning(
                    f"Invalid score value in pattern match for {model_name}"
                )

        return None

    def _extract_scores_using_pattern_matching(
        self,
        evaluation_text: str,
        model_names: list[str],
    ) -> dict[str, float]:
        scores = {}
        for model_name in model_names:
            score = self._extract_score_for_model(
                evaluation_text, model_name, SCORE_EXTRACTION_PATTERNS
            )
            if score is not None:
                scores[model_name] = score
        return scores

    def _extract_scores_with_alternative_names(
        self,
        evaluation_text: str,
        model_names: list[str],
    ) -> dict[str, float]:
        scores = {}
        numbered_mapping = {
            f"{ANONYMOUS_MODEL_PREFIX}{idx + 1}": model_name
            for idx, model_name in enumerate(model_names)
        }
        response_mapping = {
            f"{ANONYMOUS_RESPONSE_PREFIX} {idx + 1}": model_name
            for idx, model_name in enumerate(model_names)
        }
        numbered_mapping.update(response_mapping)

        alternative_names = list(numbered_mapping.keys())
        extracted = self._extract_scores_using_pattern_matching(
            evaluation_text, alternative_names
        )

        for alt_name, score in extracted.items():
            if alt_name in numbered_mapping:
                original_name = numbered_mapping[alt_name]
                if original_name not in scores:
                    scores[original_name] = score
                    self.logger.debug(
                        f"Mapped '{alt_name}' → '{original_name}' with score {score}"
                    )

        return scores
