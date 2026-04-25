import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from certamen.domain.tournament.tournament import ModelComparison

_SCORE_EPSILON = 1e-9


class RankingEngine:
    def __init__(self, comparison_instance: "ModelComparison") -> None:
        self.comp = comparison_instance
        self.logger = comparison_instance.logger

    def _handle_no_evaluation_scores(self) -> tuple[str, str]:
        self.logger.error(
            "No evaluation scores available for elimination decision. Falling back to random selection."
        )
        active_names = [
            self.comp.anon_mapping[k] for k in self.comp.active_model_keys
        ]
        self.comp.anonymizer.rng.shuffle(active_names)
        lowest = active_names[0]
        highest = (
            active_names[-1] if len(active_names) > 1 else active_names[0]
        )
        self.comp.elimination_reason = (
            "Random selection (no evaluation scores available)"
        )
        self.comp.elimination_score = None
        return lowest, highest

    def _select_leader_from_scored_models(
        self, active_scores: dict[str, float]
    ) -> str:
        self.logger.debug(
            "Selecting leader from %s scored models", len(active_scores)
        )
        max_score = max(active_scores.values())
        models_with_max = [
            name
            for name, score in active_scores.items()
            if math.isclose(score, max_score, abs_tol=_SCORE_EPSILON)
        ]
        leader = self.comp.anonymizer.rng.choice(models_with_max)
        if len(models_with_max) > 1:
            self.logger.info(
                "Tie for leader (score=%.2f): %s, randomly selected %s",
                max_score,
                models_with_max,
                leader,
            )
        else:
            self.logger.info(
                "Leader selected: %s (score=%.2f)", leader, max_score
            )
        return leader

    def _handle_unscored_models(
        self,
        unscored_models: set[str],
        active_scores: dict[str, float],
        all_active_models: set[str],
    ) -> tuple[str, str]:
        self.logger.info("Handling %s unscored models", len(unscored_models))
        lowest_model_name = self.comp.anonymizer.rng.choice(
            list(unscored_models)
        )
        self.logger.warning(
            "Models %s were not scored by any evaluator. "
            "Randomly selecting %s for elimination.",
            list(unscored_models),
            lowest_model_name,
        )
        self.comp.elimination_reason = (
            "Random selection (model was not scored by any evaluator)"
        )
        self.comp.elimination_score = None

        if not active_scores:
            remaining_models = list(all_active_models - {lowest_model_name})
            highest_model_name = (
                self.comp.anonymizer.rng.choice(remaining_models)
                if remaining_models
                else lowest_model_name
            )
            self.logger.warning(
                "No models received scores. Randomly selecting leader."
            )
        else:
            highest_model_name = self._select_leader_from_scored_models(
                active_scores
            )

        return lowest_model_name, highest_model_name

    def _select_lowest_ranked_model(
        self, active_scores: dict[str, float]
    ) -> tuple[str, float]:
        self.logger.debug(
            "Selecting lowest-ranked model from %s scores", len(active_scores)
        )
        min_score = min(active_scores.values())
        models_with_min = [
            name
            for name, score in active_scores.items()
            if math.isclose(score, min_score, abs_tol=_SCORE_EPSILON)
        ]

        if len(models_with_min) > 1:
            lowest_model_name = self.comp.anonymizer.rng.choice(
                models_with_min
            )
            self.logger.warning(
                "Tie for lowest score (%.2f): %s. "
                "Randomly selected %s for elimination.",
                min_score,
                models_with_min,
                lowest_model_name,
            )
            self.comp.elimination_reason = f"Random selection among tied models (tied at score {min_score:.2f})"
        else:
            lowest_model_name = models_with_min[0]
            self.logger.info(
                "Eliminating %s (lowest score: %.2f)",
                lowest_model_name,
                min_score,
            )
            self.comp.elimination_reason = "Lowest score in evaluation"

        self.comp.elimination_score = min_score
        return lowest_model_name, min_score

    def _select_highest_ranked_model(
        self, active_scores: dict[str, float]
    ) -> str:
        self.logger.debug(
            "Selecting highest-ranked model from %s scores", len(active_scores)
        )
        max_score = max(active_scores.values())
        models_with_max = [
            name
            for name, score in active_scores.items()
            if math.isclose(score, max_score, abs_tol=_SCORE_EPSILON)
        ]

        if len(models_with_max) > 1:
            highest_model_name = self.comp.anonymizer.rng.choice(
                models_with_max
            )
            self.logger.info(
                "Tie for highest score (%.2f): %s. Randomly selected %s as leader.",
                max_score,
                models_with_max,
                highest_model_name,
            )
        else:
            highest_model_name = models_with_max[0]
            self.logger.info(
                "Leader: %s (highest score: %.2f)",
                highest_model_name,
                max_score,
            )

        return highest_model_name

    def _resolve_same_model_tie(
        self,
        lowest_model_name: str,
        highest_model_name: str,
        active_scores: dict[str, float],
        min_score: float,
    ) -> str:
        self.logger.debug("Checking for same-model tie resolution")
        if lowest_model_name == highest_model_name and len(active_scores) > 1:
            self.logger.warning(
                "All models tied with score %.2f. Randomly selecting for elimination and leadership.",
                min_score,
            )
            other_models = [
                name
                for name in active_scores.keys()
                if name != lowest_model_name
            ]
            if other_models:
                highest_model_name = self.comp.anonymizer.rng.choice(
                    other_models
                )
            self.logger.info(
                "Randomly selecting %s to be eliminated and %s to lead.",
                lowest_model_name,
                highest_model_name,
            )
            self.comp.elimination_reason = (
                f"Random selection (all models tied at {min_score:.2f})"
            )
            self.comp.elimination_score = min_score

        return highest_model_name

    def _handle_all_models_scored(
        self, active_scores: dict[str, float]
    ) -> tuple[str, str]:
        self.logger.info(
            "All models received scores. Determining winner and loser based on scores."
        )

        lowest_model_name, min_score = self._select_lowest_ranked_model(
            active_scores
        )
        highest_model_name = self._select_highest_ranked_model(active_scores)
        highest_model_name = self._resolve_same_model_tie(
            lowest_model_name, highest_model_name, active_scores, min_score
        )

        return lowest_model_name, highest_model_name

    def _finalize_ranking_results(
        self,
        lowest_model_name: str,
        highest_model_name: str,
        active_scores: dict[str, float],
    ) -> tuple[str, str]:
        self.comp.current_leader_key = next(
            (
                key
                for key, name in self.comp.anon_mapping.items()
                if name == highest_model_name
            ),
            None,
        )

        highest_score_val = active_scores.get(highest_model_name, float("nan"))
        lowest_score_val = active_scores.get(lowest_model_name, 0.0)

        self.logger.info(
            "\nHighest-ranked model: %s with score %.2f/10",
            highest_model_name,
            highest_score_val,
            extra={"display_type": "colored_text", "color": "success"},
        )
        self.logger.info(
            "Lowest-ranked model: %s with score %.2f/10",
            lowest_model_name,
            lowest_score_val,
            extra={"display_type": "colored_text", "color": "warning"},
        )

        return lowest_model_name, highest_model_name

    def determine_lowest_and_highest_ranked_models(
        self,
    ) -> tuple[str, str]:
        if not self.comp.evaluation_scores:
            return self._handle_no_evaluation_scores()

        self.logger.info(
            "Aggregating Evaluation Scores",
            extra={"display_type": "section_header"},
        )
        aggregated_scores = self.comp.aggregator.get_aggregated_scores()

        all_active_models = {
            self.comp.anon_mapping[k] for k in self.comp.active_model_keys
        }
        active_scores = {
            k: v
            for k, v in aggregated_scores.items()
            if k in all_active_models and v is not None
        }
        unscored_models = all_active_models - set(active_scores.keys())

        if not all_active_models:
            self.logger.error("No active models remaining for ranking")
            return self._handle_no_evaluation_scores()

        if unscored_models:
            lowest, highest = self._handle_unscored_models(
                unscored_models, active_scores, all_active_models
            )
        else:
            lowest, highest = self._handle_all_models_scored(active_scores)

        return self._finalize_ranking_results(lowest, highest, active_scores)
