import statistics
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from certamen.domain.tournament.tournament import ModelComparison


class ScoreAggregator:
    def __init__(self, comparison_instance: "ModelComparison") -> None:
        self.comp = comparison_instance
        self.logger = comparison_instance.logger

    def _display_model_score(
        self,
        model_name: str,
        score: float,
        score_type: str = "Score",
    ) -> None:
        self.logger.info(
            "%s: %s %.2f/10",
            model_name,
            score_type,
            score,
            extra={"display_type": "colored_text", "color": model_name},
        )

    def _split_valid_invalid_evaluators(
        self,
    ) -> tuple[list[str], list[str]]:
        valid_evaluators = [
            evaluator
            for evaluator, evaluations in self.comp.evaluation_scores.items()
            if evaluations
        ]
        invalid_evaluators = [
            evaluator
            for evaluator, evaluations in self.comp.evaluation_scores.items()
            if not evaluations
        ]
        self.logger.info(
            "Evaluator quality check: %s valid, %s invalid",
            len(valid_evaluators),
            len(invalid_evaluators),
        )
        if invalid_evaluators:
            self.logger.warning(
                "Invalid evaluators (no scores): %s", invalid_evaluators
            )
        return valid_evaluators, invalid_evaluators

    def _finalize_model_score(
        self,
        model_display_name: str,
        scores: list[float],
    ) -> float:
        if scores:
            median = statistics.median(scores)

            if len(scores) > 1:
                variance = statistics.variance(scores)
                stdev = statistics.stdev(scores)

                self.comp.model_score_variances[model_display_name] = {
                    "variance": variance,
                    "stdev": stdev,
                    "num_judges": len(scores),
                }

                self.logger.info(
                    "  → Median score for %s: %.2f (stdev=%.2f, from %s judges)",
                    model_display_name,
                    median,
                    stdev,
                    len(scores),
                )
            else:
                self.logger.info(
                    "  → Median score for %s: %.2f (from 1 judge only)",
                    model_display_name,
                    median,
                )

            self._display_model_score(
                model_display_name, median, "Median score"
            )
            return median
        else:
            self.logger.warning(
                "No valid scores found for %s. Assigning a penalty score of 0.0.",
                model_display_name,
            )
            return 0.0

    def _normalize_judge_scores(
        self, judge_scores: dict[str, float], judge_name: str
    ) -> dict[str, float]:
        self.logger.debug(
            "Normalizing %s's scores: %s models", judge_name, len(judge_scores)
        )
        if not judge_scores or len(judge_scores) < 2:
            self.logger.debug(
                "Skipping normalization for %s: < 2 scores", judge_name
            )
            return judge_scores

        scores_list = list(judge_scores.values())
        mean_score = statistics.mean(scores_list)

        if len(set(scores_list)) == 1:
            self.logger.debug(
                "All scores identical for %s, no normalization needed",
                judge_name,
            )
            return judge_scores

        std_dev = statistics.stdev(scores_list)

        z_scores_by_model = {
            model_name: (score - mean_score) / std_dev
            for model_name, score in judge_scores.items()
        }

        z_values = list(z_scores_by_model.values())
        min_z = min(z_values)
        max_z = max(z_values)
        z_range = max_z - min_z

        if z_range > 0:
            normalized = {
                model_name: 1.0 + ((z_score - min_z) / z_range) * 9.0
                for model_name, z_score in z_scores_by_model.items()
            }
        else:
            normalized = dict.fromkeys(judge_scores.keys(), 5.5)

        self.logger.info(
            "Normalized %s's scores (mean=%.2f, std=%.2f)",
            judge_name,
            mean_score,
            std_dev,
        )

        return normalized

    def _extract_numeric_scores(
        self, raw_scores: dict[str, Any], evaluator: str
    ) -> dict[str, float]:
        self.logger.debug(
            "Extracting numeric scores from %s: %s raw scores",
            evaluator,
            len(raw_scores),
        )
        numeric_scores = {}
        failed_extractions = []
        for model_name, score in raw_scores.items():
            if isinstance(score, (int, float)):
                validated = self.comp.score_extractor.normalize_score(
                    float(score), evaluator
                )
                if validated is not None:
                    numeric_scores[model_name] = validated
                    self.logger.debug(
                        "%s: %s → %s", model_name, score, validated
                    )
                else:
                    failed_extractions.append(f"{model_name}={score}")
            else:
                failed_extractions.append(
                    f"{model_name}={type(score).__name__}"
                )
        if failed_extractions:
            self.logger.debug(
                "Failed to extract scores: %s", ", ".join(failed_extractions)
            )
        self.logger.info(
            "Extracted %s/%s scores from %s",
            len(numeric_scores),
            len(raw_scores),
            evaluator,
        )
        return numeric_scores

    def _detect_self_scoring_bias(
        self, numeric_scores: dict[str, float], evaluator: str
    ) -> float | None:
        self.logger.debug("Detecting self-scoring bias for %s", evaluator)
        if evaluator not in numeric_scores:
            self.logger.debug(
                "%s not in scores, skipping bias detection", evaluator
            )
            return None

        self_score = numeric_scores[evaluator]
        other_scores = [s for m, s in numeric_scores.items() if m != evaluator]

        if not other_scores:
            self.logger.debug("No other scores to compare for %s", evaluator)
            return None

        avg_others = statistics.mean(other_scores)
        bias = self_score - avg_others

        if bias > 1.5:
            self.logger.warning(
                "Potential self-scoring bias detected: %s gave itself "
                "%.1f vs %.1f avg to others (bias: +%.1f)",
                evaluator,
                self_score,
                avg_others,
                bias,
            )
        elif bias < -1.5:
            self.logger.info(
                "%s scored itself lower than others: "
                "%.1f vs %.1f avg (bias: %.1f)",
                evaluator,
                self_score,
                avg_others,
                bias,
            )
        else:
            self.logger.debug(
                "%s self-score: %.1f, others avg: %.1f (bias: %+.1f)",
                evaluator,
                self_score,
                avg_others,
                bias,
            )

        return bias

    def _normalize_evaluator_scores(
        self, evaluator: str
    ) -> tuple[float | None, dict[str, float]] | None:
        self.logger.debug("Normalizing scores from evaluator: %s", evaluator)
        raw_scores = self.comp.evaluation_scores[evaluator]
        if not isinstance(raw_scores, dict):
            self.logger.warning(
                "%s has non-dict scores, skipping normalization", evaluator
            )
            return None

        numeric_scores = self._extract_numeric_scores(raw_scores, evaluator)
        if not numeric_scores:
            self.logger.warning(
                "No numeric scores extracted from %s", evaluator
            )
            return None

        bias = self._detect_self_scoring_bias(numeric_scores, evaluator)
        normalized = self._normalize_judge_scores(numeric_scores, evaluator)
        if bias is not None:
            self.logger.info(
                "%s normalization complete with bias=%+.2f", evaluator, bias
            )
            return bias, normalized

        self.logger.info(
            "%s normalization complete, no bias detected", evaluator
        )
        return None, normalized

    def _log_self_scoring_bias_summary(
        self, self_scoring_biases: dict[str, float]
    ) -> None:
        self.comp.self_scoring_biases = self_scoring_biases
        self.logger.info(
            "Self-scoring bias summary: %s models scored themselves",
            len(self_scoring_biases),
        )
        for model, bias in sorted(
            self_scoring_biases.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            bias_str = f"{bias:+.2f}"
            if abs(bias) > 1.5:
                self.logger.info("   • %s: %s (significant)", model, bias_str)
            else:
                self.logger.debug("   • %s: %s", model, bias_str)

    def _collect_scores_for_model(
        self,
        model_display_name: str,
        normalized_evaluation_scores: dict[str, dict[str, float]],
        valid_evaluators: list[str],
    ) -> list[float]:
        scores: list[float] = []
        self.logger.info("Collecting scores for %s:", model_display_name)

        for evaluator in valid_evaluators:
            if evaluator not in normalized_evaluation_scores:
                continue

            normalized_scores = normalized_evaluation_scores[evaluator]
            if model_display_name not in normalized_scores:
                continue

            norm_score = normalized_scores[model_display_name]
            scores.append(norm_score)

            is_self_score = evaluator == model_display_name
            score_type = " (self-score)" if is_self_score else ""
            self.logger.info(
                "  - %s gave %s a normalized score of %.2f%s",
                evaluator,
                model_display_name,
                norm_score,
                score_type,
            )

        return scores

    def _aggregate_peer_review_scores(self) -> dict[str, float]:
        self.logger.info("Aggregating scores from peer-review format.")
        aggregated_scores: dict[str, float] = {}

        valid_evaluators, invalid_evaluators = (
            self._split_valid_invalid_evaluators()
        )

        if invalid_evaluators:
            self.logger.warning(
                "%s evaluator(s) provided invalid evaluations and were excluded: %s",
                len(invalid_evaluators),
                ", ".join(invalid_evaluators),
            )

        normalized_evaluation_scores = {}
        self_scoring_biases = {}

        for evaluator in valid_evaluators:
            result = self._normalize_evaluator_scores(evaluator)
            if result is None:
                continue

            bias, normalized_scores = result
            if bias is not None:
                self_scoring_biases[evaluator] = bias

            if normalized_scores:
                normalized_evaluation_scores[evaluator] = normalized_scores

        if self_scoring_biases:
            self._log_self_scoring_bias_summary(self_scoring_biases)

        for model_display_name in [
            self.comp.anon_mapping[k] for k in self.comp.active_model_keys
        ]:
            scores = self._collect_scores_for_model(
                model_display_name,
                normalized_evaluation_scores,
                valid_evaluators,
            )
            aggregated_scores[model_display_name] = self._finalize_model_score(
                model_display_name, scores
            )

        return aggregated_scores

    def _aggregate_judge_scores(self) -> dict[str, float]:
        self.logger.info("Processing scores from single-judge format.")
        result_scores: dict[str, float] = {}
        evaluator_name = next(iter(self.comp.all_evaluations.keys()), "judge")

        for model_name, score in self.comp.evaluation_scores.items():
            if isinstance(score, (int, float)):
                score_float = float(score)
                normalized_score = self.comp.score_extractor.normalize_score(
                    score_float, evaluator_name
                )
                if normalized_score is not None:
                    self._display_model_score(
                        model_name, normalized_score, "Judge score"
                    )
                    result_scores[model_name] = normalized_score
                else:
                    self.logger.warning(
                        "Judge gave %s an invalid score of %s (rejected)",
                        model_name,
                        score_float,
                    )
            else:
                self.logger.warning(
                    "Skipping non-numeric score for %s: %s (type: %s)",
                    model_name,
                    score,
                    type(score).__name__,
                )

        return result_scores

    def get_aggregated_scores(self) -> dict[str, float]:
        if not self.comp.evaluation_scores:
            return {}
        first_score_value = next(iter(self.comp.evaluation_scores.values()))
        is_peer_review = isinstance(first_score_value, dict)

        if is_peer_review:
            return self._aggregate_peer_review_scores()
        else:
            return self._aggregate_judge_scores()
